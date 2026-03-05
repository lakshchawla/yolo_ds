#include <cuda_runtime_api.h>
#include <glib.h>
#include <gst/gst.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include "gst-nvmessage.h"
#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"

#define MAX_DISPLAY_LEN 64
#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2
#define OSD_PROCESS_MODE 1
#define OSD_DISPLAY_TEXT 1
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

gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person", "RoadSign" };

static gboolean PERF_MODE = FALSE;

static gboolean
bus_call(GstBus* bus, GstMessage* msg, gpointer data)
{
    GMainLoop* loop = (GMainLoop*)data;
    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
        g_print("End of stream\n");
        g_main_loop_quit(loop);
        break;
    case GST_MESSAGE_WARNING: {
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
    case GST_MESSAGE_ERROR: {
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
cb_newpad(GstElement* decodebin, GstPad* decoder_src_pad, gpointer data)
{
    GstCaps* caps = gst_pad_get_current_caps(decoder_src_pad);
    if (!caps) {
        caps = gst_pad_query_caps(decoder_src_pad, NULL);
    }
    const GstStructure* str = gst_caps_get_structure(caps, 0);
    const gchar* name = gst_structure_get_name(str);

    GstElement* source_bin = (GstElement*)data;
    GstCapsFeatures* features = gst_caps_get_features(caps, 0);

    if (!strncmp(name, "video", 5)) {
        if (gst_caps_features_contains(features, GST_CAPS_FEATURES_NVMM)) {
            GstPad* bin_ghost_pad = gst_element_get_static_pad(source_bin, "src");

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
decodebin_child_added(GstChildProxy* child_proxy, GObject* object,
    gchar* name, gpointer user_data)
{
    g_print("Decodebin child added: %s\n", name);
    if (g_strrstr(name, "decodebin") == name) {
        g_signal_connect(G_OBJECT(object), "child-added",
            G_CALLBACK(decodebin_child_added), user_data);
    }
    if (g_strrstr(name, "source") == name) {
        g_object_set(G_OBJECT(object), "drop-on-latency", true, NULL);
    }
}

static GstElement*
create_source_bin(guint index, gchar* uri)
{
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

int main(int argc, char* argv[])
{
    GMainLoop* loop = NULL;
    GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL,
               *queue1, *queue2, *queue3, *queue4, *queue5, *nvvidconv = NULL,
               *nvosd = NULL, *tiler = NULL, *nvdslogger = NULL;

    GstElement *tracker = NULL;

    GstBus* bus = NULL;
    guint bus_watch_id;
    GstPad* tiler_src_pad = NULL;
    guint i = 0, num_sources = 0;
    guint tiler_rows, tiler_columns;
    guint pgie_batch_size;
    gboolean yaml_config = FALSE;
    NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;

    PERF_MODE = g_getenv("NVDS_TEST3_PERF_MODE") && !g_strcmp0(g_getenv("NVDS_TEST3_PERF_MODE"), "1");

    int current_device = -1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    /* Check input arguments */
    if (argc < 2) {
        g_printerr("Usage: %s <yml file>\n", argv[0]);
        g_printerr("OR: %s <uri1> [uri2] ... [uriN] \n", argv[0]);
        return -1;
    }

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    yaml_config = (g_str_has_suffix(argv[1], ".yml") || g_str_has_suffix(argv[1], ".yaml"));

    if (yaml_config) {
        RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1],
            "primary-gie"));
    }

    pipeline = gst_pipeline_new("dstest3-pipeline");

    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

    if (!pipeline || !streammux) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }
    gst_bin_add(GST_BIN(pipeline), streammux);

    GList* src_list = NULL;

    if (yaml_config) {

        RETURN_ON_PARSER_ERROR(nvds_parse_source_list(&src_list, argv[1], "source-list"));

        GList* temp = src_list;
        while (temp) {
            num_sources++;
            temp = temp->next;
        }
        g_list_free(temp);
    } else {
        num_sources = argc - 1;
    }

    for (i = 0; i < num_sources; i++) {
        GstPad *sinkpad, *srcpad;
        gchar pad_name[16] = {};

        GstElement* source_bin = NULL;
        if (g_str_has_suffix(argv[1], ".yml") || g_str_has_suffix(argv[1], ".yaml")) {
            g_print("Now playing : %s\n", (char*)(src_list)->data);
            source_bin = create_source_bin(i, (char*)(src_list)->data);
        } else {
            source_bin = create_source_bin(i, argv[i + 1]);
        }
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

        if (yaml_config) {
            src_list = src_list->next;
        }
    }

    if (yaml_config) {
        g_list_free(src_list);
    }

    pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    tracker = gst_element_factory_make("nvtracker", "tracker");

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
        /* Finally render the osd output */
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

    if (!pgie || !nvdslogger || !tiler || !nvvidconv || !nvosd || !sink) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    if (!tracker) {
        g_printerr("Tracker could not be created. Exiting.\n");
    }


    RETURN_ON_PARSER_ERROR(nvds_parse_streammux(streammux, argv[1], "streammux"));

    RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie, argv[1], "primary-gie"));

    g_object_get(G_OBJECT(pgie), "batch-size", &pgie_batch_size, NULL);
    if (pgie_batch_size != num_sources) {
        g_printerr("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
            pgie_batch_size, num_sources);
        g_object_set(G_OBJECT(pgie), "batch-size", num_sources, NULL);
    }

    g_object_set(G_OBJECT(tracker), "tracker-width", 640, NULL);
    g_object_set(G_OBJECT(tracker), "tracker-height", 384, NULL);
    g_object_set(G_OBJECT(tracker), "gpu_id", 0, NULL);
    g_object_set(G_OBJECT(tracker), "ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so", NULL);
    g_object_set(G_OBJECT(tracker), "ll-config-file", "config_tracker_NvDCF_perf.yml", NULL);

    RETURN_ON_PARSER_ERROR(nvds_parse_osd(nvosd, argv[1], "osd"));

    tiler_rows = (guint)sqrt(num_sources);
    tiler_columns = (guint)ceil(1.0 * num_sources / tiler_rows);
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

    gst_bin_add_many(GST_BIN(pipeline), queue1, pgie, queue2, tracker, nvdslogger, tiler,
        queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL);

    if (!gst_element_link_many(streammux, queue1, pgie, queue2, nvdslogger, tiler,
            queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL)) {
        g_printerr("Elements could not be linked. Exiting.\n");
        return -1;
    }

    // tiler_src_pad = gst_element_get_static_pad(pgie, "src");
    // if (!tiler_src_pad)
    //     g_print("Unable to get src pad\n");
    // else
    //     gst_pad_add_probe(tiler_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
    //         tiler_src_pad_buffer_probe, NULL, NULL);
    // gst_object_unref(tiler_src_pad);

    if (yaml_config) {
        g_print("Using file: %s\n", argv[1]);
    } else {
        g_print("Now playing:");
        for (i = 0; i < num_sources; i++) {
            g_print(" %s,", argv[i + 1]);
        }
        g_print("\n");
    }
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