import json, time, threading, argparse, signal, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
import paho.mqtt.client as mqtt

SUB_TOPIC       = "reid/detections"   # inbound from DS
PUB_ID_TOPIC    = "reid/global_ids"   # resolved IDs → DS

TRACKLET_TIMEOUT_SEC  = 2.0    # declare tracklet done after this silence
TRACKLET_MAX_FRAMES   = 15     # or after this many frames
MIN_BBOX_W            = 2     # filter tiny crops before averaging
MIN_BBOX_H            = 5
MATCH_COST_THRESHOLD  = 0.30   # cosine distance — below = same person
EMA_ALPHA             = 0.90   # gallery update momentum

tracklet_lock  = threading.Lock()
gallery_lock   = threading.Lock()


active_tracklets: dict = {}
gallery: dict = {}
next_global_id = 1000

mqtt_client_ref = None   # set after connect


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Both vectors must already be L2-normalized."""
    return float(1.0 - np.dot(a, b))


def ingest_detection(sensor_id: str, track_id: int,
                     bbox: list, confidence: float,
                     reid_vector: np.ndarray):
    key = f"{sensor_id}_{track_id}"
    now = time.time()

    with tracklet_lock:
        if key not in active_tracklets:
            active_tracklets[key] = {
                "sensor_id":   sensor_id,
                "track_id":    track_id,
                "vectors":     [],
                "bboxes":      [],
                "last_seen":   now,
                "frame_count": 0,
            }

        t = active_tracklets[key]
        t["last_seen"]   = now
        t["frame_count"] += 1

        # quality gate: skip tiny crops
        w, h = bbox[2], bbox[3]
        if w >= MIN_BBOX_W and h >= MIN_BBOX_H:
            t["vectors"].append(reid_vector)
            t["bboxes"].append(bbox)

        # flush if max frames reached
        if t["frame_count"] >= TRACKLET_MAX_FRAMES:
            tracklet = active_tracklets.pop(key)
            threading.Thread(target=resolve_tracklet,
                             args=(tracklet,), daemon=True).start()


def build_tracklet_vector(vectors: list) -> np.ndarray | None:
    if not vectors:
        return None
    avg = np.mean(np.stack(vectors), axis=0)
    return l2_normalize(avg)


# ── Module 2 + 3 + 4: Match & update gallery ──────────────────────────────────
def resolve_tracklet(tracklet: dict):
    global next_global_id

    vec = build_tracklet_vector(tracklet["vectors"])
    if vec is None:
        return   # all frames were too small

    sensor_id = tracklet["sensor_id"]
    track_id  = tracklet["track_id"]
    now       = time.time()

    with gallery_lock:
        if not gallery:
            # first ever person — just register
            gid = next_global_id; next_global_id += 1
            gallery[gid] = {
                "vector":      vec,
                "last_sensor": sensor_id,
                "last_time":   now,
            }
            publish_global_id(sensor_id, track_id, gid, is_new=True)
            print(f"[gallery] NEW  Global_ID={gid}  (first person)")
            return

        # build cost matrix: 1 tracklet × N gallery entries
        gids    = list(gallery.keys())
        g_vecs  = np.stack([gallery[g]["vector"] for g in gids])  # N×D
        costs   = 1.0 - g_vecs.dot(vec)                           # N cosine distances

        # Hungarian on a 1×N matrix
        cost_matrix = costs.reshape(1, -1)
        _, col_ind  = linear_sum_assignment(cost_matrix)
        best_col    = col_ind[0]
        best_cost   = cost_matrix[0, best_col]
        best_gid    = gids[best_col]

        if best_cost < MATCH_COST_THRESHOLD:
            # ── accept match, EMA update ──
            old_vec = gallery[best_gid]["vector"]
            new_vec = l2_normalize(EMA_ALPHA * old_vec + (1 - EMA_ALPHA) * vec)
            gallery[best_gid]["vector"]      = new_vec
            gallery[best_gid]["last_sensor"] = sensor_id
            gallery[best_gid]["last_time"]   = now
            publish_global_id(sensor_id, track_id, best_gid, is_new=False)
            print(f"[gallery] MATCH  track={track_id}@{sensor_id} "
                  f"→ Global_ID={best_gid}  cost={best_cost:.4f}")
        else:
            # ── new identity ──
            gid = next_global_id; next_global_id += 1
            gallery[gid] = {
                "vector":      vec,
                "last_sensor": sensor_id,
                "last_time":   now,
            }
            publish_global_id(sensor_id, track_id, gid, is_new=True)
            print(f"[gallery] NEW   track={track_id}@{sensor_id} "
                  f"→ Global_ID={gid}  cost={best_cost:.4f}")


def publish_global_id(sensor_id: str, track_id: int,
                      global_id: int, is_new: bool):
    if not mqtt_client_ref:
        return
    msg = json.dumps({
        "sensor_id": sensor_id,
        "local_track_id": track_id,
        "global_id": global_id,
        "is_new_identity": is_new,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    mqtt_client_ref.publish(PUB_ID_TOPIC, msg, qos=1)


# ── Tracklet timeout reaper (runs in background) ───────────────────────────────
def tracklet_reaper():
    while True:
        time.sleep(0.5)
        now = time.time()
        expired = []
        with tracklet_lock:
            for key, t in active_tracklets.items():
                if now - t["last_seen"] > TRACKLET_TIMEOUT_SEC:
                    expired.append(key)
            for key in expired:
                tracklet = active_tracklets.pop(key)
                threading.Thread(target=resolve_tracklet,
                                 args=(tracklet,), daemon=True).start()


# ── MQTT callbacks ─────────────────────────────────────────────────────────────
def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except json.JSONDecodeError as e:
        print(f"[WARN] bad JSON: {e}")
        return

    sensor_id  = payload["sensor_id"]
    track_id   = int(payload["local_track_id"])
    bbox       = payload["bbox"]
    confidence = float(payload["det_confidence"])
    vector     = np.array(payload["reid_vector"], dtype=np.float32)

    ingest_detection(sensor_id, track_id, bbox, confidence, vector)


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        client.subscribe(SUB_TOPIC, qos=1)
        print(f"[MQTT] Connected — subscribed to '{SUB_TOPIC}'")
        print(f"[MQTT] Publishing global IDs to  '{PUB_ID_TOPIC}'")
    else:
        print(f"[MQTT] Connection refused rc={reason_code}")


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    print(f"[MQTT] Disconnected rc={reason_code}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    global mqtt_client_ref

    parser = argparse.ArgumentParser(description="ReID backend matching engine")
    parser.add_argument("--broker",    default="localhost")
    parser.add_argument("--port",      default=1883, type=int)
    parser.add_argument("--client-id", default="reid-backend")
    args = parser.parse_args()

    client = mqtt.Client(
        client_id=args.client_id,
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )
    mqtt_client_ref   = client
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_message    = on_message

    def _shutdown(sig, frame):
        print("\n[INFO] Shutting down backend...")
        client.disconnect()
        sys.exit(0)
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # start the reaper before connecting
    threading.Thread(target=tracklet_reaper, daemon=True).start()

    client.connect(args.broker, args.port, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()