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
#include <list>
#include <mutex>
#include <cmath>
#include <numeric>
#include <vector>
#include <algorithm>
// #include <Eigen/Dense>

#include "gst-nvmessage.h"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvds_analytics_meta.h"
// #ifndef PLATFORM_TEGRA
#include "gst-nvmessage.h"
#include "nvds_yml_parser.h"
#include<array>
#include <chrono>
#include <unordered_map>

#include <string>
#include <chrono>

#define MAX_DISPLAY_LEN 64
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080
#define MUXER_BATCH_TIMEOUT_USEC 40000
#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

guint PERF_MODE = 0;

gfloat EID_MATCH_THRESHOLD = 0.30;
gfloat REID_EMA_ALPHA = 0.90;
gdouble REID_LOST_TRACK_TTL = 3.0;
gint REID_MIN_FRAMES_TO_REGISTER = 10;
gint REID_MIN_BBOX_W = 40; //add depth logic as well (if applicatble)
gint REID_MIN_BBOX_H = 80;
gint REID_N_CLUSTERS = 4;
gfloat CLUST_INIT_THRESH = 0.10;


using Vec       = std::vector<float>;
using Clock     = std::chrono::steady_clock;
using TimePoint = Clock::time_point;


#define RETURN_ON_PARSER_ERROR(parse_expr)                    \
if (NVDS_YAML_PARSER_SUCCESS != parse_expr) {                 \
g_printerr("Error in parsing configuration file.\n");         \
return -1;                                                    \
}

static gboolean
bus_call(GstBus* bus, GstMessage* msg, gpointer data)
{
    GMainLoop* loop = (GMainLoop*)data;
    switch (GST_MESSAGE_TYPE(msg))
    {
    case GST_MESSAGE_EOS:
        g_print("End of stream\n");
        g_main_loop_quit(loop);
        break;
    case GST_MESSAGE_WARNING:
        {
            gchar* debug = NULL;
            GError* error = NULL;
            gst_message_parse_warning(msg, &error, &debug);
            g_printerr("WARNING from element %s: %s\n",
                       GST_OBJECT_NAME(msg->src), error->message);
            g_free(debug);
            g_printerr("Warning: %s\n", error->message);
            g_error_free(error);
            break;
        }
    case GST_MESSAGE_ERROR:
        {
            gchar* debug = NULL;
            GError* error = NULL;
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
    case GST_MESSAGE_ELEMENT:
        {
            /* Special handling: If one camera in a multi-stream setup dies (EOS),
             * we can catch it here without killing the whole app. */
            if (gst_nvmessage_is_stream_eos(msg))
            {
                guint stream_id = 0;
                if (gst_nvmessage_parse_stream_eos(msg, &stream_id))
                {
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
cb_newpad(GstElement* decodebin, GstPad* decoder_src_pad, gpointer data)
{
    GstCaps* caps = gst_pad_get_current_caps(decoder_src_pad);
    if (!caps)
    {
        caps = gst_pad_query_caps(decoder_src_pad, NULL);
    }
    const GstStructure* str = gst_caps_get_structure(caps, 0);
    const gchar* name = gst_structure_get_name(str);

    GstElement* source_bin = (GstElement*)data;
    GstCapsFeatures* features = gst_caps_get_features(caps, 0);

    if (!strncmp(name, "video", 5))
    {
        if (gst_caps_features_contains(features, GST_CAPS_FEATURES_NVMM))
        {
            GstPad* bin_ghost_pad = gst_element_get_static_pad(source_bin, "src");

            if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad),
                                          decoder_src_pad))
            {
                g_printerr("Failed to link decoder src pad to source bin ghost pad\n");
            }
            gst_object_unref(bin_ghost_pad);
        }
        else
        {
            g_printerr("Error: Decodebin did not pick nvidia decoder plugin.\n");
        }
    }
}

static void
decodebin_child_added(GstChildProxy* child_proxy, GObject* object,
                      gchar* name, gpointer user_data)
{
    g_print("Decodebin child added: %s\n", name);
    if (g_strrstr(name, "decodebin") == name)
    {
        g_signal_connect(G_OBJECT(object), "child-added",
                         G_CALLBACK(decodebin_child_added), user_data);
    }
    if (g_strrstr(name, "source") == name)
    {
        g_object_set(G_OBJECT(object), "drop-on-latency", true, NULL);
    }
}

static GstElement*
create_source_bin(guint index, gchar* uri)
{
    g_print("%s", uri);

    GstElement *bin = NULL, *uri_decode_bin = NULL;
    gchar bin_name[16] = {};

    g_snprintf(bin_name, 15, "source-bin-%02d", index);
    bin = gst_bin_new(bin_name);

    if (PERF_MODE)
    {
        uri_decode_bin = gst_element_factory_make("nvurisrcbin", "uri-decode-bin");
        g_object_set(G_OBJECT(uri_decode_bin), "file-loop", TRUE, NULL);
        g_object_set(G_OBJECT(uri_decode_bin), "cudadec-memtype", 0, NULL);
    }
    else
    {
        uri_decode_bin = gst_element_factory_make("uridecodebin", "uri-decode-bin");
    }

    if (!bin || !uri_decode_bin)
    {
        g_printerr("One element in source bin could not be created.\n");
        return NULL;
    }

    g_object_set(G_OBJECT(uri_decode_bin), "uri", uri, NULL);

    // std::string uri = g_object_get(G_OBJECT(uri_decode_bin), ("uri");

    g_signal_connect(G_OBJECT(uri_decode_bin), "pad-added",
                     G_CALLBACK(cb_newpad), bin);
    g_signal_connect(G_OBJECT(uri_decode_bin), "child-added",
                     G_CALLBACK(decodebin_child_added), bin);

    gst_bin_add(GST_BIN(bin), uri_decode_bin);

    if (!gst_element_add_pad(bin, gst_ghost_pad_new_no_target("src", GST_PAD_SRC)))
    {
        g_printerr("Failed to add ghost pad in source bin\n");
        return NULL;
    }

    return bin;
}


static float vec_dot(const Vec &a, const Vec &b)
{
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
}

static float vec_norm(const Vec &v)
{
    return std::sqrt(vec_dot(v, v));
}

static void l2_normalise(Vec &v)
{
    float n = vec_norm(v);
    if (n > 1e-9f)
        for (float &x : v) x /= n;
}

static float cosine_dist(const Vec &a, const Vec &b)
{
    return std::max(0.0f, 1.0f - vec_dot(a, b));
}

static float min_dist_between_sets(const std::vector<Vec> &A,
                                   const std::vector<Vec> &B)
{
    float best = 1.0f;
    for (const Vec &a : A)
        for (const Vec &b : B)
            best = std::min(best, cosine_dist(a, b));
    return best;
}

static double seconds_since(const TimePoint &tp)
{
    return std::chrono::duration<double>(Clock::now() - tp).count();
}

static void welford_update(Vec &avg, const Vec &val, int count)
{
    for (size_t i = 0; i < avg.size(); i++)
        avg[i] += (val[i] - avg[i]) / static_cast<float>(count);
}

class ClusterFeature
{
public:
    ClusterFeature() = default;

    void update(const Vec &vec, float weight = 1.0f)
    {
        if (centroids_.empty()) {
            centroids_.push_back(vec);
            sizes_.push_back(weight);
            return;
        }

        int   nearest = 0;
        float min_d   = cosine_dist(vec, centroids_[0]);
        for (int j = 1; j < (int)centroids_.size(); j++) {
            float d = cosine_dist(vec, centroids_[j]);
            if (d < min_d) { min_d = d; nearest = j; }
        }

        if (min_d > CLUST_INIT_THRESH && (int)centroids_.size() < REID_N_CLUSTERS) {
            centroids_.push_back(vec);
            sizes_.push_back(weight);
        } else {
            float s = sizes_[nearest];
            float w = weight / (s + weight);
            Vec  &c = centroids_[nearest];
            for (size_t i = 0; i < c.size(); i++)
                c[i] += (vec[i] - c[i]) * w;
            sizes_[nearest] += weight;
            l2_normalise(c);
        }
    }

    float best_distance_to(const ClusterFeature &other) const
    {
        if (centroids_.empty() || other.centroids_.empty()) return 1.0f;
        return min_dist_between_sets(centroids_, other.centroids_);
    }

    void merge_from(const ClusterFeature &other)
    {
        for (int i = 0; i < (int)other.centroids_.size(); i++)
            update(other.centroids_[i], other.sizes_[i]);
    }

    bool empty() const { return centroids_.empty(); }

private:
    std::vector<Vec>   centroids_;
    std::vector<float> sizes_;
};

inline double get_current_time_sec() {
    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> seconds = now.time_since_epoch();
    return seconds.count();
}


struct Tracklets
{
    gint8 dsid;
    gint8 gid;
    ClusterFeature cluster;
    std::vector<float> avg_vec;
    bool avg_valid = false;
    gint frame_count;
    TimePoint last_seen_time;

    Tracklets(int dsid, gint8 gid) : dsid(dsid), gid(gid), frame_count(0) {}

    void add_embedding(const Vec &vec, int bw, int bh)
    {
        if (bw < REID_MIN_BBOX_W || bh < REID_MIN_BBOX_H) return;

        cluster.update(vec);

        if (!avg_valid) {
            avg_vec   = vec;
            avg_valid = true;
        } else {
            frame_count++;
            welford_update(avg_vec, vec, frame_count);
            l2_normalise(avg_vec);
        }
        frame_count = std::max(frame_count, 1);
        last_seen_time = Clock::now();
    }

    bool is_gallery_worthy() const
    {
        return frame_count >= REID_MIN_FRAMES_TO_REGISTER && !cluster.empty();
    }

};

static GstPadProbeReturn
reid_pad_buffer_probe(GstPad* pad, GstPadProbeInfo* info, gpointer user_data)
{
    GstBuffer* buffer = (GstBuffer*)info->data;
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list;
         l_frame != NULL; l_frame = l_frame->next)

    {
        NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)l_frame->data;


        for (NvDsMetaList* l_obj = frame_meta->obj_meta_list;

             l_obj != NULL; l_obj = l_obj->next)

        {
            NvDsObjectMeta* obj_meta = (NvDsObjectMeta*)l_obj->data;
            for (NvDsMetaList* l_user = obj_meta->obj_user_meta_list;
                 l_user != NULL; l_user = l_user->next)
            {

                NvDsUserMeta* user_meta = (NvDsUserMeta*)l_user->data;
                if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) continue;

                NvDsInferTensorMeta* tensor_meta = (NvDsInferTensorMeta*)user_meta->user_meta_data;
                float* embedding_vector = (float*)tensor_meta->out_buf_ptrs_host[0];

                // _get_embeddings - returning embedding vector
            }
        }
    }
    return GST_PAD_PROBE_OK;
}


int main(int argc, char* argv[])
{
    GMainLoop* loop = NULL;
    GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL,
               *nvvidconv = NULL,
               *nvosd = NULL, *tiler = NULL, *nvdslogger = NULL, *nvtracker = NULL;
    // GstElement *nvdsanalytics = NULL, *nvdsananalytics = NULL;

    GstElement *queue1, *queue2, *queue3, *queue4, *queue5, *queue6;
    GstElement* sgie1 = NULL;

    GstBus* bus = NULL;
    GstPad* osd_sink_pad = NULL;
    GstPad* reid_sgie_pad = NULL;
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

    if (argc < 2)
    {
        g_printerr("Usage: %s <app_config.yml>\n", argv[0]);
        return -1;
    }

    yaml_config = (g_str_has_suffix(argv[1], ".yml") || g_str_has_suffix(argv[1], ".yaml"));

    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1], "primary-gie"));
    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1], "secondary-gie-1"));

    pipeline = gst_pipeline_new("dstest3-pipeline");
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

    gst_bin_add(GST_BIN(pipeline), streammux);
    GList* src_list = NULL;

    RETURN_ON_PARSER_ERROR(nvds_parse_source_list(&src_list, argv[1], "source-list"));
    GList* temp = src_list;

    while (temp)
    {
        num_sources++;
        temp = temp->next;
    }

    for (i = 0; i < num_sources; i++)
    {
        GstPad *sinkpad, *srcpad;
        gchar pad_name[16] = {};

        GstElement* source_bin = NULL;
        g_print("Now playing : %s\n", (char*)(src_list)->data);
        source_bin = create_source_bin(i, (char*)(src_list)->data);

        if (!source_bin)
        {
            g_printerr("Failed to create source bin. Exiting.\n");
            return -1;
        }

        gst_bin_add(GST_BIN(pipeline), source_bin);

        g_snprintf(pad_name, 15, "sink_%u", i);
        sinkpad = gst_element_request_pad_simple(streammux, pad_name);
        if (!sinkpad)
        {
            g_printerr("Streammux request sink pad failed. Exiting.\n");
            return -1;
        }

        srcpad = gst_element_get_static_pad(source_bin, "src");
        if (!srcpad)
        {
            g_printerr("Failed to get src pad of source bin. Exiting.\n");
            return -1;
        }

        if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
        {
            g_printerr("Failed to link source bin to stream muxer. Exiting.\n");
            return -1;
        }

        gst_object_unref(srcpad);
        gst_object_unref(sinkpad);

        src_list = src_list->next;
    }

    g_list_free_full(g_list_first(temp), g_free);

    // GstElement* custom_plugin = gst_element_factory_make("nvdsvideotemplate", "cutom-plugin");

    pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    sgie1 = gst_element_factory_make("nvinfer", "secondary-nvinference-engine-1");
    nvtracker = gst_element_factory_make("nvtracker", "tracker");

    queue1 = gst_element_factory_make("queue", "queue1");
    queue2 = gst_element_factory_make("queue", "queue2");
    queue3 = gst_element_factory_make("queue", "queue3");
    queue4 = gst_element_factory_make("queue", "queue4");
    queue5 = gst_element_factory_make("queue", "queue5");
    queue6 = gst_element_factory_make("queue", "queue6");

    nvdslogger = gst_element_factory_make("nvdslogger", "nvdslogger");
    tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");
    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

    if (PERF_MODE)
    {
        sink = gst_element_factory_make("fakesink", "nvvideo-renderer");
    }
    else
    {
        if (prop.integrated)
        {
            sink = gst_element_factory_make("nv3dsink", "nv3d-sink"); // Jetson
        }
        else
        {
#ifdef __aarch64__
            sink = gst_element_factory_make("nv3dsink", "nvvideo-renderer");
#else
            sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer"); // Desktop GPU
#endif
        }
    }

    if (!pgie || !sgie1 || !nvdslogger || !tiler || !nvvidconv || !nvosd || !sink || !nvtracker)
    {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    RETURN_ON_PARSER_ERROR(nvds_parse_streammux(streammux, argv[1], "streammux"));
    RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie, argv[1], "primary-gie"));
    RETURN_ON_PARSER_ERROR(nvds_parse_gie(sgie1, argv[1], "secondary-gie-1"));

    g_object_get(G_OBJECT(pgie), "batch-size", &pgie_batch_size, NULL);
    g_object_get(G_OBJECT(sgie1), "batch-size", &pgie_batch_size, NULL);
    if (pgie_batch_size != num_sources)
    {
        g_printerr("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
                   pgie_batch_size, num_sources);
        g_object_set(G_OBJECT(pgie), "batch-size", num_sources, NULL);
    }

    RETURN_ON_PARSER_ERROR(nvds_parse_tracker(nvtracker, argv[1], "tracker"));
    RETURN_ON_PARSER_ERROR(nvds_parse_osd(nvosd, argv[1], "osd"));

    g_object_set(G_OBJECT(nvosd), "display-text", 1, "process-mode", 1, NULL);

    tiler_rows = (guint)sqrt(num_sources);
    tiler_columns = (guint)ceil(1.0 * num_sources / tiler_rows);
    g_object_set(G_OBJECT(tiler), "rows", tiler_rows, "columns", tiler_columns, NULL);

    RETURN_ON_PARSER_ERROR(nvds_parse_tiler(tiler, argv[1], "tiler"));


    if (PERF_MODE)
    {
        RETURN_ON_PARSER_ERROR(nvds_parse_fake_sink(sink, argv[1], "sink"));
    }
    else if (prop.integrated)
    {
        RETURN_ON_PARSER_ERROR(nvds_parse_3d_sink(sink, argv[1], "sink"));
    }
    else
    {
#ifdef __aarch64__
        RETURN_ON_PARSER_ERROR(nvds_parse_3d_sink(sink, argv[1], "sink"));
#else
        RETURN_ON_PARSER_ERROR(nvds_parse_egl_sink(sink, argv[1], "sink"));
#endif
    }

    if (PERF_MODE)
    {
        if (prop.integrated)
        {
            g_object_set(G_OBJECT(streammux), "nvbuf-memory-type", 4, NULL);
        }
        else
        {
            g_object_set(G_OBJECT(streammux), "nvbuf-memory-type", 2, NULL);
        }
    }

    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    gst_bin_add_many(GST_BIN(pipeline), queue1, pgie, queue2, nvtracker, queue3, nvdslogger, queue4, nvvidconv,
                     queue5, nvosd, queue6, sink, NULL);
    if (!gst_element_link_many(streammux, queue1, pgie, queue2, nvtracker, queue3, nvdslogger, queue4, nvvidconv,
                               queue5, nvosd, queue6, sink, NULL))
    {
        g_printerr("Elements could not be linked. Exiting.\n");
        return -1;
    }

    // reid_sgie_pad = gst_element_get_static_pad(sink, "sink");
    // if (!reid_sgie_pad)
    // {
    //     g_printerr("Pad could not be linked. Exiting.\n");
    //     return -1;
    // }
    // else
    // {
    //     gst_pad_add_probe(reid_sgie_pad, GST_PAD_PROBE_TYPE_BUFFER, reid_pad_buffer_probe, NULL, NULL);
    //     gst_object_unref(reid_sgie_pad);
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