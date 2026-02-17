#include "tools/replay/framereader.h"
#include "tools/replay/nvdec_decoder.h"

extern "C" {
#include <libavutil/imgutils.h>
}

namespace {

enum AVPixelFormat get_nvdec_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
  enum AVPixelFormat *hw_pix_fmt = reinterpret_cast<enum AVPixelFormat *>(ctx->opaque);
  for (const enum AVPixelFormat *p = pix_fmts; *p != -1; p++) {
    if (*p == *hw_pix_fmt) return *p;
  }
  rWarning("NVDEC hw pixel format not found, falling back to software decode");
  *hw_pix_fmt = AV_PIX_FMT_NONE;
  return AV_PIX_FMT_YUV420P;
}

}  // namespace

NvdecVideoDecoder::NvdecVideoDecoder() {
  av_frame_ = av_frame_alloc();
  hw_frame_ = av_frame_alloc();
}

NvdecVideoDecoder::~NvdecVideoDecoder() {
  if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);
  if (decoder_ctx) avcodec_free_context(&decoder_ctx);
  av_frame_free(&av_frame_);
  av_frame_free(&hw_frame_);
}

bool NvdecVideoDecoder::open(AVCodecParameters *codecpar, bool hw_decoder) {
  const AVCodec *decoder = nullptr;
  using_v4l2_ = false;

  if (hw_decoder) {
    // Try Jetson-specific V4L2 hardware decoders first (NVDEC via v4l2)
    if (codecpar->codec_id == AV_CODEC_ID_H264) {
      decoder = avcodec_find_decoder_by_name("h264_nvv4l2dec");
    } else if (codecpar->codec_id == AV_CODEC_ID_HEVC) {
      decoder = avcodec_find_decoder_by_name("hevc_nvv4l2dec");
    }

    if (decoder) {
      rInfo("Using Jetson V4L2 NVDEC decoder: %s", decoder->name);
      using_v4l2_ = true;
    } else {
      rWarning("Jetson V4L2 NVDEC decoder not found, trying CUDA hwaccel fallback");
    }
  }

  // Fallback: use generic decoder (will attach CUDA hwaccel below if hw_decoder requested)
  if (!decoder) {
    decoder = avcodec_find_decoder(codecpar->codec_id);
    if (!decoder) {
      rError("No suitable decoder found for codec id %d", codecpar->codec_id);
      return false;
    }
  }

  decoder_ctx = avcodec_alloc_context3(decoder);
  if (!decoder_ctx || avcodec_parameters_to_context(decoder_ctx, codecpar) != 0) {
    rError("Failed to allocate or initialize NVDEC codec context");
    return false;
  }

  width = (decoder_ctx->width + 3) & ~3;
  height = decoder_ctx->height;

  // For V4L2 decoders, no additional hw device setup needed -- the decoder handles it
  // For generic decoders with CUDA hwaccel, set up hw device context
  if (!using_v4l2_ && hw_decoder) {
    const AVCodecHWConfig *config = nullptr;
    for (int i = 0; (config = avcodec_get_hw_config(decoder, i)) != nullptr; i++) {
      if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
          config->device_type == AV_HWDEVICE_TYPE_CUDA) {
        hw_pix_fmt = config->pix_fmt;
        break;
      }
    }

    if (config) {
      int ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
      if (ret < 0) {
        rWarning("Failed to create CUDA hw device context: %d. Falling back to CPU decoding.", ret);
        hw_pix_fmt = AV_PIX_FMT_NONE;
      } else {
        decoder_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
        decoder_ctx->opaque = &hw_pix_fmt;
        decoder_ctx->get_format = get_nvdec_hw_format;
        rInfo("NVDEC: using CUDA hwaccel for decoding");
      }
    } else {
      rWarning("CUDA hardware config not found for codec. Falling back to CPU decoding.");
    }
  }

  if (avcodec_open2(decoder_ctx, decoder, nullptr) < 0) {
    rError("Failed to open NVDEC codec");
    return false;
  }

  rInfo("NVDEC decoder opened: %dx%d (v4l2=%d)", width, height, using_v4l2_);
  return true;
}

bool NvdecVideoDecoder::decode(FrameReader *reader, int idx, VisionBuf *buf) {
  int current_idx = idx;
  if (idx != reader->prev_idx + 1) {
    // Seek to the nearest key frame
    for (int i = idx; i >= 0; --i) {
      if (reader->packets_info[i].flags & AV_PKT_FLAG_KEY) {
        current_idx = i;
        break;
      }
    }

    auto pos = reader->packets_info[current_idx].pos;
    int ret = avformat_seek_file(reader->input_ctx, 0, pos, pos, pos, AVSEEK_FLAG_BYTE);
    if (ret < 0) {
      rError("NVDEC: Failed to seek to byte position %lld: %d", (long long)pos, AVERROR(ret));
      return false;
    }
    avcodec_flush_buffers(decoder_ctx);
  }
  reader->prev_idx = idx;

  AVPacket pkt;
  while (av_read_frame(reader->input_ctx, &pkt) >= 0) {
    // Skip non-video packets
    if (pkt.stream_index != reader->video_stream_idx_) {
      av_packet_unref(&pkt);
      continue;
    }

    AVFrame *frame = decodeFrame(&pkt);
    av_packet_unref(&pkt);
    if (!frame) {
      rError("NVDEC: Failed to decode frame at index %d", current_idx);
      return false;
    }

    if (current_idx++ == idx) {
      return copyBuffer(frame, buf);
    }
  }
  rError("NVDEC: Failed to find frame at index %d", idx);
  return false;
}

AVFrame *NvdecVideoDecoder::decodeFrame(AVPacket *pkt) {
  int ret = avcodec_send_packet(decoder_ctx, pkt);
  if (ret < 0) {
    rError("NVDEC: Error sending packet for decoding: %d", ret);
    return nullptr;
  }

  ret = avcodec_receive_frame(decoder_ctx, av_frame_);
  if (ret != 0) {
    rError("NVDEC: avcodec_receive_frame error: %d", ret);
    return nullptr;
  }

  // If using CUDA hwaccel, transfer frame from GPU to CPU
  if (hw_pix_fmt != AV_PIX_FMT_NONE && av_frame_->format == hw_pix_fmt) {
    if (av_hwframe_transfer_data(hw_frame_, av_frame_, 0) < 0) {
      rError("NVDEC: Error transferring frame data from GPU to CPU");
      return nullptr;
    }
    return hw_frame_;
  }

  // V4L2 decoders and CPU decoders return frames directly
  return av_frame_;
}

bool NvdecVideoDecoder::copyBuffer(AVFrame *f, VisionBuf *buf) {
  // On Jetson, V4L2 NVDEC decoders output NV12 natively.
  // CUDA hwaccel also outputs NV12 after transfer.
  // In both cases, data[0] = Y plane, data[1] = interleaved UV plane.
  if (f->format == AV_PIX_FMT_NV12) {
    for (int i = 0; i < height / 2; i++) {
      memcpy(buf->y + (i * 2 + 0) * buf->stride, f->data[0] + (i * 2 + 0) * f->linesize[0], width);
      memcpy(buf->y + (i * 2 + 1) * buf->stride, f->data[0] + (i * 2 + 1) * f->linesize[0], width);
      memcpy(buf->uv + i * buf->stride, f->data[1] + i * f->linesize[1], width);
    }
  } else if (f->format == AV_PIX_FMT_NV21) {
    // NV21 is the same layout but with V,U instead of U,V -- still interleaved, just swapped.
    // Jetson V4L2 may output NV21 in some configurations. Swap chroma bytes.
    for (int i = 0; i < height / 2; i++) {
      memcpy(buf->y + (i * 2 + 0) * buf->stride, f->data[0] + (i * 2 + 0) * f->linesize[0], width);
      memcpy(buf->y + (i * 2 + 1) * buf->stride, f->data[0] + (i * 2 + 1) * f->linesize[0], width);
      // Swap UV pairs: NV21 (VUVU) -> NV12 (UVUV)
      const uint8_t *src = f->data[1] + i * f->linesize[1];
      uint8_t *dst = buf->uv + i * buf->stride;
      for (int j = 0; j < width; j += 2) {
        dst[j] = src[j + 1];      // U
        dst[j + 1] = src[j];      // V
      }
    }
  } else if (f->format == AV_PIX_FMT_YUV420P) {
    // Software decode fallback: I420 -> NV12 conversion without libyuv
    // Y plane: copy directly
    for (int i = 0; i < height; i++) {
      memcpy(buf->y + i * buf->stride, f->data[0] + i * f->linesize[0], width);
    }
    // UV plane: interleave U and V
    int half_width = width / 2;
    for (int i = 0; i < height / 2; i++) {
      const uint8_t *u_row = f->data[1] + i * f->linesize[1];
      const uint8_t *v_row = f->data[2] + i * f->linesize[2];
      uint8_t *uv_dst = buf->uv + i * buf->stride;
      for (int j = 0; j < half_width; j++) {
        uv_dst[j * 2] = u_row[j];
        uv_dst[j * 2 + 1] = v_row[j];
      }
    }
  } else {
    rError("NVDEC: Unsupported pixel format %d for copyBuffer", f->format);
    return false;
  }
  return true;
}
