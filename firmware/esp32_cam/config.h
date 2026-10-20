#pragma once

#include <stdint.h>
#include "secrets.h"

// Thay bằng domain public trong Render Dashboard, không thêm path ở cuối.
static constexpr char RENDER_BASE_URL[] = "https://render-yolo-web.onrender.com";

static constexpr uint8_t CAMERA_JPEG_QUALITY = 65;
static constexpr uint32_t UPLOAD_PAUSE_MS = 250;
static constexpr uint32_t HTTP_TIMEOUT_MS = 60000;
