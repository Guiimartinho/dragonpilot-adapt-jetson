#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
}

#include "system/loggerd/encoder/encoder.h"
#include "system/loggerd/loggerd.h"

// Hardware-accelerated NVENC encoder for Jetson AGX Xavier.
// Accepts NV12 input directly from VisionBuf, avoiding the
// NV12->I420 conversion that the software FfmpegEncoder performs.
class NvencEncoder : public VideoEncoder {
public:
  NvencEncoder(const EncoderInfo &encoder_info, int in_width, int in_height);
  ~NvencEncoder();
  int encode_frame(VisionBuf* buf, VisionIpcBufExtra *extra);
  void encoder_open();
  void encoder_close();

private:
  int segment_num = -1;
  int counter = 0;
  bool is_open = false;

  const AVCodec *codec = nullptr;
  AVCodecContext *codec_ctx = nullptr;
  AVFrame *frame = nullptr;
  std::vector<uint8_t> downscale_buf;
};
