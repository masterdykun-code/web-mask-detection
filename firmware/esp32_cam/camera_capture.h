#pragma once

#include <stddef.h>
#include <stdint.h>

bool initializeCamera();
bool captureJpeg(uint8_t **jpegBuffer, size_t *jpegLength, int64_t *capturedAtMs);
void releaseJpeg(uint8_t *jpegBuffer);
