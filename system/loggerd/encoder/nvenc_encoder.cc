#include "system/loggerd/encoder/nvenc_encoder.h"

#include <fcntl.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

#define __STDC_CONSTANT_MACROS

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}

// Only needed when downscaling is required (in_size != out_size)
#include "third_party/libyuv/include/libyuv.h"

#include "common/swaglog.h"
#include "common/util.h"

const int env_debug_encoder = (getenv("DEBUG_ENCODER") != NULL) ? atoi(getenv("DEBUG_ENCODER")) : 0;

NvencEncoder::NvencEncoder(const EncoderInfo &encoder_info, int in_width, int in_height)
    : VideoEncoder(encoder_info, in_width, in_height) {

  // Try Jetson-specific h264_nvmpi first (L4T / Jetson Multimedia API FFmpeg wrapper),
  // then fall back to the generic NVIDIA h264_nvenc codec.
  codec = avcodec_find_encoder_by_name("h264_nvmpi");
  if (!codec) {
    LOGW("h264_nvmpi codec not found, falling back to h264_nvenc");
    codec = avcodec_find_encoder_by_name("h264_nvenc");
  }
  if (!codec) {
    LOGE("Neither h264_nvmpi nor h264_nvenc codec found! Falling back to software h264");
    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
  }
  assert(codec);

  frame = av_frame_alloc();
  assert(frame);

  // NV12 format: Y plane + interleaved UV plane.
  // This avoids the NV12->I420 conversion that FfmpegEncoder does.
  frame->format = AV_PIX_FMT_NV12;
  frame->width = out_width;
  frame->height = out_height;
  frame->linesize[0] = out_width;       // Y stride
  frame->linesize[1] = out_width;       // UV stride (interleaved, same width as Y)

  if (in_width != out_width || in_height != out_height) {
    // Pre-allocate all buffers for downscaling to avoid per-frame heap allocation
    downscale_buf.resize(out_width * out_height * 3 / 2);
    src_i420_buf.resize(in_width * in_height * 3 / 2);
    dst_i420_buf.resize(out_width * out_height * 3 / 2);
  }
}

NvencEncoder::~NvencEncoder() {
  encoder_close();
  av_frame_free(&frame);
}

void NvencEncoder::encoder_open() {
  codec_ctx = avcodec_alloc_context3(codec);
  assert(codec_ctx);

  codec_ctx->width = frame->width;
  codec_ctx->height = frame->height;
  codec_ctx->pix_fmt = AV_PIX_FMT_NV12;
  codec_ctx->time_base = (AVRational){ 1, encoder_info.fps };

  // Retrieve the configured bitrate and GOP size from encoder settings
  auto settings = encoder_info.get_settings(in_width);
  codec_ctx->bit_rate = settings.bitrate;
  codec_ctx->gop_size = settings.gop_size;
  codec_ctx->max_b_frames = settings.b_frames;

  AVDictionary *opts = NULL;

  // Detect which codec we resolved and set appropriate options
  std::string codec_name(codec->name);

  if (codec_name == "h264_nvmpi") {
    // h264_nvmpi (Jetson L4T) specific options
    // Profile: high for better compression at automotive bitrates
    av_dict_set(&opts, "profile", "high", 0);
    // Use CBR for consistent bitrate during driving recording
    av_dict_set(&opts, "rc", "cbr", 0);

  } else if (codec_name == "h264_nvenc") {
    // NVENC-specific options optimized for automotive recording on Jetson
    av_dict_set(&opts, "preset", "llhp", 0);        // low-latency high-performance
    av_dict_set(&opts, "profile", "high", 0);
    av_dict_set(&opts, "rc", "cbr", 0);              // constant bitrate for consistent file sizes
    av_dict_set(&opts, "delay", "0", 0);             // minimize encoding latency
    av_dict_set(&opts, "zerolatency", "1", 0);       // no reordering delay

  } else {
    // Software fallback: use fast settings
    av_dict_set(&opts, "preset", "ultrafast", 0);
    av_dict_set(&opts, "tune", "zerolatency", 0);
  }

  int err = avcodec_open2(codec_ctx, codec, &opts);
  av_dict_free(&opts);
  if (err < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(err, errbuf, sizeof(errbuf));
    LOGE("avcodec_open2 failed for %s: %s", codec->name, errbuf);
    assert(false);
  }

  LOGW("NvencEncoder opened with codec %s, %dx%d @ %d bps, gop=%d",
       codec->name, codec_ctx->width, codec_ctx->height,
       (int)codec_ctx->bit_rate, codec_ctx->gop_size);

  is_open = true;
  segment_num++;
  counter = 0;
}

void NvencEncoder::encoder_close() {
  if (!is_open) return;

  // Flush the encoder by sending a NULL frame
  avcodec_send_frame(codec_ctx, NULL);
  AVPacket pkt = {};
  while (avcodec_receive_packet(codec_ctx, &pkt) == 0) {
    av_packet_unref(&pkt);
  }

  avcodec_free_context(&codec_ctx);
  codec_ctx = nullptr;
  is_open = false;
}

int NvencEncoder::encode_frame(VisionBuf* buf, VisionIpcBufExtra *extra) {
  if (buf->width != (size_t)this->in_width || buf->height != (size_t)this->in_height) {
    LOGE("NvencEncoder: buffer dimensions mismatch: got %zux%zu, expected %dx%d",
         buf->width, buf->height, this->in_width, this->in_height);
    return -1;
  }

  // VisionBuf provides NV12 data: buf->y (Y plane), buf->uv (interleaved UV plane), buf->stride
  uint8_t *src_y = buf->y;
  uint8_t *src_uv = buf->uv;
  int src_stride_y = buf->stride;
  int src_stride_uv = buf->stride;

  if (downscale_buf.size() > 0) {
    // Need to downscale: NV12->I420 at input size, scale, then I420->NV12 at output size.
    // Uses pre-allocated buffers to avoid per-frame heap allocation in the hot path.
    uint8_t *si_y = src_i420_buf.data();
    uint8_t *si_u = si_y + in_width * in_height;
    uint8_t *si_v = si_u + (in_width / 2) * (in_height / 2);

    // NV12 -> I420
    libyuv::NV12ToI420(src_y, src_stride_y,
                       src_uv, src_stride_uv,
                       si_y, in_width,
                       si_u, in_width / 2,
                       si_v, in_width / 2,
                       in_width, in_height);

    uint8_t *di_y = dst_i420_buf.data();
    uint8_t *di_u = di_y + out_width * out_height;
    uint8_t *di_v = di_u + (out_width / 2) * (out_height / 2);

    // I420 scale
    libyuv::I420Scale(si_y, in_width,
                      si_u, in_width / 2,
                      si_v, in_width / 2,
                      in_width, in_height,
                      di_y, out_width,
                      di_u, out_width / 2,
                      di_v, out_width / 2,
                      out_width, out_height,
                      libyuv::kFilterNone);

    // I420 -> NV12 into downscale_buf
    uint8_t *out_y = downscale_buf.data();
    uint8_t *out_uv = out_y + out_width * out_height;

    libyuv::I420ToNV12(di_y, out_width,
                       di_u, out_width / 2,
                       di_v, out_width / 2,
                       out_y, out_width,
                       out_uv, out_width,
                       out_width, out_height);

    frame->data[0] = out_y;
    frame->data[1] = out_uv;
    frame->linesize[0] = out_width;
    frame->linesize[1] = out_width;
  } else {
    // No downscaling needed: pass NV12 data directly from VisionBuf
    frame->data[0] = src_y;
    frame->data[1] = src_uv;
    frame->linesize[0] = src_stride_y;
    frame->linesize[1] = src_stride_uv;
  }

  frame->pts = counter * 50 * 1000; // 50ms per frame (20 fps)

  int ret = counter;

  int err = avcodec_send_frame(codec_ctx, frame);
  if (err < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(err, errbuf, sizeof(errbuf));
    LOGE("avcodec_send_frame error %d: %s", err, errbuf);
    ret = -1;
  }

  AVPacket pkt = {};
  pkt.data = NULL;
  pkt.size = 0;
  while (ret >= 0) {
    err = avcodec_receive_packet(codec_ctx, &pkt);
    if (err == AVERROR_EOF) {
      break;
    } else if (err == AVERROR(EAGAIN)) {
      // Encoder might need a few frames on startup to get started. Keep going
      ret = 0;
      break;
    } else if (err < 0) {
      char errbuf[AV_ERROR_MAX_STRING_SIZE];
      av_strerror(err, errbuf, sizeof(errbuf));
      LOGE("avcodec_receive_packet error %d: %s", err, errbuf);
      ret = -1;
      break;
    }

    if (env_debug_encoder) {
      printf("%20s got %8d bytes flags %8x idx %4d id %8d\n",
             encoder_info.publish_name, pkt.size, pkt.flags, counter, extra->frame_id);
    }

    publisher_publish(segment_num, counter, *extra,
      (pkt.flags & AV_PKT_FLAG_KEY) ? V4L2_BUF_FLAG_KEYFRAME : 0,
      kj::arrayPtr<capnp::byte>(pkt.data, (size_t)0), // TODO: get the header
      kj::arrayPtr<capnp::byte>(pkt.data, pkt.size));

    counter++;
  }
  av_packet_unref(&pkt);
  return ret;
}
