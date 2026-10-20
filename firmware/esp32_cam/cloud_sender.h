#pragma once

#include <Arduino.h>
#include <stddef.h>
#include <stdint.h>

struct UploadResult {
  int httpStatus;
  uint32_t roundTripMs;
  uint32_t sequence;
  String responseBody;
};

bool uploadJpegToRender(
    const uint8_t *jpegBuffer,
    size_t jpegLength,
    int64_t capturedAtMs,
    UploadResult *result);
bool reportUploadRoundTrip(uint32_t sequence, uint32_t roundTripMs);
