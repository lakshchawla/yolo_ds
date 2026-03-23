//
// Created by lakshh on 3/6/26.
//

#include <cuda_runtime_api.h>
#include <glib.h>
#include <iostream>
#include <gst/gst.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sstream>
#include <list>
#include <deque>
#include <mutex>
#include <memory>

#include <fstream>
#include <iomanip>
#include <filesystem>

#include "gst-nvmessage.h"
#include "gstnvdsmeta.h"
#include "nvds_analytics_meta.h"
// #ifndef PLATFORM_TEGRA
#include "gst-nvmessage.h"
#include "nvds_yml_parser.h"

#include <chrono>

#include <unordered_map>
#include <queue>
// #include <memory>

#define MAX_DISPLAY_LEN 64
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080
#define MUXER_BATCH_TIMEOUT_USEC 40000
#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

#define RETURN_ON_PARSER_ERROR(parse_expr)                    \
if (NVDS_YAML_PARSER_SUCCESS != parse_expr) {             \
g_printerr("Error in parsing configuration file.\n"); \
return -1;                                            \
}

static gboolean PERF_MODE = FALSE;

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_WARNING: {
            gchar *debug = NULL;
            GError *error = NULL;
            gst_message_parse_warning(msg, &error, &debug);
            g_printerr("WARNING from element %s: %s\n",
                       GST_OBJECT_NAME(msg->src), error->message);
            g_free(debug);
            g_printerr("Warning: %s\n", error->message);
            g_error_free(error);
            break;
        }
        case GST_MESSAGE_ERROR: {
            gchar *debug = NULL;
            GError *error = NULL;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("ERROR from element %s: %s\n",
                       GST_OBJECT_NAME(msg->src), error->message);
            if (debug)
                g_printerr("Error details: %s\n", debug);
            g_free(debug);
            g_error_free(error);
            g_main_loop_quit(loop);
            break;
        }
        case GST_MESSAGE_ELEMENT: {
            /* Special handling: If one camera in a multi-stream setup dies (EOS),
             * we can catch it here without killing the whole app. */
            if (gst_nvmessage_is_stream_eos(msg)) {
                guint stream_id = 0;
                if (gst_nvmessage_parse_stream_eos(msg, &stream_id)) {
                    g_print("Got EOS from stream %d\n", stream_id);
                }
            }
            break;
        }
        default:
            break;
    }
    return TRUE;
}

static void
cb_newpad(GstElement *decodebin, GstPad *decoder_src_pad, gpointer data) {
    GstCaps *caps = gst_pad_get_current_caps(decoder_src_pad);
    if (!caps) {
        caps = gst_pad_query_caps(decoder_src_pad, NULL);
    }
    const GstStructure *str = gst_caps_get_structure(caps, 0);
    const gchar *name = gst_structure_get_name(str);

    GstElement *source_bin = (GstElement *) data;
    GstCapsFeatures *features = gst_caps_get_features(caps, 0);

    if (!strncmp(name, "video", 5)) {
        if (gst_caps_features_contains(features, GST_CAPS_FEATURES_NVMM)) {
            GstPad *bin_ghost_pad = gst_element_get_static_pad(source_bin, "src");

            if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad),
                                          decoder_src_pad)) {
                g_printerr("Failed to link decoder src pad to source bin ghost pad\n");
            }
            gst_object_unref(bin_ghost_pad);
        } else {
            g_printerr("Error: Decodebin did not pick nvidia decoder plugin.\n");
        }
    }
}

static void
decodebin_child_added(GstChildProxy *child_proxy, GObject *object,
                      gchar *name, gpointer user_data) {
    g_print("Decodebin child added: %s\n", name);
    if (g_strrstr(name, "decodebin") == name) {
        g_signal_connect(G_OBJECT(object), "child-added",
                         G_CALLBACK(decodebin_child_added), user_data);
    }
    if (g_strrstr(name, "source") == name) {
        g_object_set(G_OBJECT(object), "drop-on-latency", true, NULL);
    }
}

static GstElement *
create_source_bin(guint index, gchar *uri) {
    GstElement *bin = NULL, *uri_decode_bin = NULL;
    gchar bin_name[16] = {};

    g_snprintf(bin_name, 15, "source-bin-%02d", index);
    bin = gst_bin_new(bin_name);

    if (PERF_MODE) {
        uri_decode_bin = gst_element_factory_make("nvurisrcbin", "uri-decode-bin");
        g_object_set(G_OBJECT(uri_decode_bin), "file-loop", TRUE, NULL);
        g_object_set(G_OBJECT(uri_decode_bin), "cudadec-memtype", 0, NULL);
    } else {
        uri_decode_bin = gst_element_factory_make("uridecodebin", "uri-decode-bin");
    }

    if (!bin || !uri_decode_bin) {
        g_printerr("One element in source bin could not be created.\n");
        return NULL;
    }

    g_object_set(G_OBJECT(uri_decode_bin), "uri", uri, NULL);

    g_signal_connect(G_OBJECT(uri_decode_bin), "pad-added",
                     G_CALLBACK(cb_newpad), bin);
    g_signal_connect(G_OBJECT(uri_decode_bin), "child-added",
                     G_CALLBACK(decodebin_child_added), bin);

    gst_bin_add(GST_BIN(bin), uri_decode_bin);

    if (!gst_element_add_pad(bin, gst_ghost_pad_new_no_target("src", GST_PAD_SRC))) {
        g_printerr("Failed to add ghost pad in source bin\n");
        return NULL;
    }

    return bin;
}

struct Point {
    gfloat x;
    gfloat y;
};

struct EntityState {
    gfloat speed;
    gfloat direction;
};

typedef std::unordered_map<guint64, Point> FrameSnapshot;
typedef std::unordered_map<guint64, EntityState> PerFrameDynamics;
typedef std::deque<std::unique_ptr<PerFrameDynamics>> TemporalFeed;

TemporalFeed g_temporal_feed;
FrameSnapshot g_prev_frame_data; // Persists across probe calls to compare with "current"
std::mutex g_data_mutex;

const gint8 FPS = 30;
const gint8 INTERVAL = 0;
const gint8 BANDWIDTH_SECS = 3 ;
const gint MAX_QUEUE_SIZE = (FPS * BANDWIDTH_SECS);
gint8 initial_frame_check = 0;

void save_history_to_disk(guint stream_id, const TemporalFeed& feed, const std::string& base_path) {
    if (feed.empty()) return;

    std::filesystem::create_directories(base_path);
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << base_path << "/stream_" << stream_id << "_"
       << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S") << ".csv";
    std::string filename = ss.str();

    std::ofstream file(filename);

    if (file.is_open()) {
        // 3. Header mapping directly to your EntityState struct
        file << "frame_offset,object_id,speed,direction_radians\n";

        int frame_offset = 0;

        // 4. Iterate through the deque
        for (const auto& dynamics_ptr : feed) {
            // Check if the unique_ptr is valid before dereferencing
            if (dynamics_ptr) {
                // Iterate through the unordered_map inside the pointer
                for (const auto& [obj_id, state] : *dynamics_ptr) {
                    file << frame_offset << ","
                         << obj_id << ","
                         << state.speed << ","
                         << state.direction << "\n";
                }
            }
            frame_offset++;
        }
        file.close();
    } else {
        g_printerr("Failed to open file for writing: %s\n", filename.c_str());
    }
}


static GstPadProbeReturn
osd_analytics_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buffer = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);

    std::lock_guard<std::mutex> lock(g_data_mutex);

    // 1. Prepare maps for this specific batch/frame
    FrameSnapshot curr_frame_data;
    auto curr_dynamics = std::make_unique<PerFrameDynamics>();

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            guint64 id = obj_meta->object_id;

            Point curr_pt;
            curr_pt.x = obj_meta->rect_params.left + (obj_meta->rect_params.width / 2);
            curr_pt.y = obj_meta->rect_params.top + obj_meta->rect_params.height;
            curr_frame_data[id] = curr_pt;

            // 2. Euclidean Distance Calculation (Velocity Vector)
            if (g_prev_frame_data.count(id)) {
                Point prev_pt = g_prev_frame_data[id];
                float dx = curr_pt.x - prev_pt.x;
                float dy = curr_pt.y - prev_pt.y;

                EntityState state;
                state.speed = std::sqrt(dx * dx + dy * dy); // Magnitude
                state.direction = std::atan2(dy, dx);       // Angle in radians

                g_print("%d, %f, %f \n", id, state.speed, state.direction);

                (*curr_dynamics)[id] = state;
            }
        }
    }

    g_temporal_feed.push_back(std::move(curr_dynamics));

    if (g_temporal_feed.size() > MAX_QUEUE_SIZE) {
        save_history_to_disk(0, g_temporal_feed, "/home/lakshh/Desktop/metadata_logs");
        g_temporal_feed.pop_front();
    }

    // 4. Preserve current coordinates for the next frame's comparison
    g_prev_frame_data = std::move(curr_frame_data);

    return GST_PAD_PROBE_OK;
}

int main(int argc, char *argv[]) {
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL,
            *queue1, *queue2, *queue3, *queue4, *queue5, *nvvidconv = NULL,
            *nvosd = NULL, *tiler = NULL, *nvdslogger = NULL, *nvtracker = NULL;
    // GstElement *nvdsanalytics = NULL, *nvdsananalytics = NULL;
    GstBus *bus = NULL;
    GstPad *osd_sink_pad = NULL;
    guint bus_watch_id;
    guint i = 0, num_sources = 0;
    guint tiler_rows, tiler_columns;
    guint pgie_batch_size;
    gboolean yaml_config = FALSE;
    NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;

    PERF_MODE = g_getenv("NVDS_TEST3_PERF_MODE") && !g_strcmp0(g_getenv("NVDS_TEST3_PERF_MODE"), "1");

    int current_device = 0;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    if (argc < 2) {
        g_printerr("Usage: %s <app_config.yml>\n", argv[0]);
        return -1;
    }

    yaml_config = (g_str_has_suffix(argv[1], ".yml") || g_str_has_suffix(argv[1], ".yaml"));
    if (!yaml_config) {
        g_printerr("Error: Only YAML config is supported.\n");
        return -1;
    }

    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1], "primary-gie"));
    pipeline = gst_pipeline_new("dstest3-pipeline");
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

    if (!pipeline || !streammux) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    gst_bin_add(GST_BIN(pipeline), streammux);
    GList *src_list = NULL;
    RETURN_ON_PARSER_ERROR(nvds_parse_source_list(&src_list, argv[1], "source-list"));
    GList *temp = src_list;
    while (temp) {
        num_sources++;
        temp = temp->next;
    }

    for (i = 0; i < num_sources; i++) {
        GstPad *sinkpad, *srcpad;
        gchar pad_name[16] = {};

        GstElement *source_bin = NULL;
        g_print("Now playing : %s\n", (char *) (src_list)->data);
        source_bin = create_source_bin(i, (char *) (src_list)->data);

        if (!source_bin) {
            g_printerr("Failed to create source bin. Exiting.\n");
            return -1;
        }

        gst_bin_add(GST_BIN(pipeline), source_bin);

        g_snprintf(pad_name, 15, "sink_%u", i);
        sinkpad = gst_element_request_pad_simple(streammux, pad_name);
        if (!sinkpad) {
            g_printerr("Streammux request sink pad failed. Exiting.\n");
            return -1;
        }

        srcpad = gst_element_get_static_pad(source_bin, "src");
        if (!srcpad) {
            g_printerr("Failed to get src pad of source bin. Exiting.\n");
            return -1;
        }

        if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
            g_printerr("Failed to link source bin to stream muxer. Exiting.\n");
            return -1;
        }

        gst_object_unref(srcpad);
        gst_object_unref(sinkpad);

        src_list = src_list->next;
    }

    g_list_free_full(g_list_first(temp), g_free);

    pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    nvtracker = gst_element_factory_make("nvtracker", "tracker");

    queue1 = gst_element_factory_make("queue", "queue1");
    queue2 = gst_element_factory_make("queue", "queue2");
    queue3 = gst_element_factory_make("queue", "queue3");
    queue4 = gst_element_factory_make("queue", "queue4");
    queue5 = gst_element_factory_make("queue", "queue5");

    nvdslogger = gst_element_factory_make("nvdslogger", "nvdslogger");
    tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");
    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

    if (PERF_MODE) {
        sink = gst_element_factory_make("fakesink", "nvvideo-renderer");
    } else {
        if (prop.integrated) {
            sink = gst_element_factory_make("nv3dsink", "nv3d-sink"); // Jetson
        } else {
#ifdef __aarch64__
            sink = gst_element_factory_make("nv3dsink", "nvvideo-renderer");
#else
            sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer"); // Desktop GPU
#endif
        }
    }

    if (!pgie || !nvdslogger || !tiler || !nvvidconv || !nvosd || !sink || !nvtracker) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    RETURN_ON_PARSER_ERROR(nvds_parse_streammux(streammux, argv[1], "streammux"));
    RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie, argv[1], "primary-gie"));

    g_object_get(G_OBJECT(pgie), "batch-size", &pgie_batch_size, NULL);
    if (pgie_batch_size != num_sources) {
        g_printerr("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
                   pgie_batch_size, num_sources);
        g_object_set(G_OBJECT(pgie), "batch-size", num_sources, NULL);
    }

    // YAML Parser handles the tracker logic cleanly here
    RETURN_ON_PARSER_ERROR(nvds_parse_tracker(nvtracker, argv[1], "tracker"));
    RETURN_ON_PARSER_ERROR(nvds_parse_osd(nvosd, argv[1], "osd"));

    g_object_set(G_OBJECT(nvosd), "display-text", 1, "process-mode", 1, NULL);

    tiler_rows = (guint) sqrt(num_sources);
    tiler_columns = (guint) ceil(1.0 * num_sources / tiler_rows);
    g_object_set(G_OBJECT(tiler), "rows", tiler_rows, "columns", tiler_columns, NULL);

    RETURN_ON_PARSER_ERROR(nvds_parse_tiler(tiler, argv[1], "tiler"));

    if (PERF_MODE) {
        RETURN_ON_PARSER_ERROR(nvds_parse_fake_sink(sink, argv[1], "sink"));
    } else if (prop.integrated) {
        RETURN_ON_PARSER_ERROR(nvds_parse_3d_sink(sink, argv[1], "sink"));
    } else {
#ifdef __aarch64__
        RETURN_ON_PARSER_ERROR(nvds_parse_3d_sink(sink, argv[1], "sink"));
#else
        RETURN_ON_PARSER_ERROR(nvds_parse_egl_sink(sink, argv[1], "sink"));
#endif
    }

    if (PERF_MODE) {
        if (prop.integrated) {
            g_object_set(G_OBJECT(streammux), "nvbuf-memory-type", 4, NULL);
        } else {
            g_object_set(G_OBJECT(streammux), "nvbuf-memory-type", 2, NULL);
        }
    }

    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    gst_bin_add_many(GST_BIN(pipeline), queue1, pgie, queue2, nvtracker, nvdslogger, tiler,
                     queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL);

    // Linking order strictly maintained
    if (!gst_element_link_many(streammux, queue1, pgie, queue2, nvtracker, nvdslogger, tiler,
                               queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL)) {
        g_printerr("Elements could not be linked. Exiting.\n");
        return -1;
    }

    // osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
    // if (!osd_sink_pad) {
    //     g_printerr("Unable Pad!");
    //     return -1;
    // } else {
    //     gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
    //                       osd_analytics_pad_buffer_probe, NULL, NULL);
    //     gst_object_unref(osd_sink_pad);
    // }

    g_print("Using file: %s\n", argv[1]);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    g_print("Running...\n");
    g_main_loop_run(loop);

    g_print("Returned, stopping playback\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    g_print("Deleting pipeline\n");
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    return 0;
}
