#pragma once

#include "msgq/visionipc/visionbuf.h"
#include "tools/replay/util.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
}

class FrameReader;
class VideoDecoder;

class NvdecVideoDecoder : public VideoDecoder {
public:
  NvdecVideoDecoder();
  ~NvdecVideoDecoder() override;
  bool open(AVCodecParameters *codecpar, bool hw_decoder) override;
  bool decode(FrameReader *reader, int idx, VisionBuf *buf) override;

private:
  AVFrame *decodeFrame(AVPacket *pkt);
  bool copyBuffer(AVFrame *f, VisionBuf *buf);

  AVFrame *av_frame_ = nullptr;
  AVFrame *hw_frame_ = nullptr;
  AVCodecContext *decoder_ctx = nullptr;
  AVBufferRef *hw_device_ctx = nullptr;
  AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;
  bool using_v4l2_ = false;
};
