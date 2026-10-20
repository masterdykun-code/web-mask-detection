#include "camera_capture.h"

#include <Arduino.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#include "board_config.h"
#include "config.h"
#include "esp_camera.h"
#include "img_converters.h"

bool initializeCamera() {
  camera_config_t config;
  memset(&config, 0, sizeof(config));

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_VGA;
  config.pixel_format = PIXFORMAT_RGB565;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = CAMERA_JPEG_QUALITY;
  config.fb_count = 2;

  const esp_err_t error = esp_camera_init(&config);
  if (error != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", error);
    return false;
  }

  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor) {
    sensor->set_vflip(sensor, 0);
    sensor->set_hmirror(sensor, 0);
    sensor->set_brightness(sensor, 0);
    sensor->set_contrast(sensor, 0);
    sensor->set_saturation(sensor, 0);
  }
  return true;
}

bool captureJpeg(uint8_t **jpegBuffer, size_t *jpegLength, int64_t *capturedAtMs) {
  if (!jpegBuffer || !jpegLength || !capturedAtMs) return false;
  *jpegBuffer = nullptr;
  *jpegLength = 0;
  *capturedAtMs = 0;

  camera_fb_t *frame = esp_camera_fb_get();
  if (!frame) {
    Serial.println("Camera capture failed");
    return false;
  }

  timeval now{};
  if (gettimeofday(&now, nullptr) == 0 && now.tv_sec > 1700000000) {
    *capturedAtMs = static_cast<int64_t>(now.tv_sec) * 1000 + now.tv_usec / 1000;
  }

  const bool converted = frame2jpg(frame, CAMERA_JPEG_QUALITY, jpegBuffer, jpegLength);
  esp_camera_fb_return(frame);
  if (!converted || !*jpegBuffer || *jpegLength == 0) {
    Serial.println("RGB565 to JPEG conversion failed");
    releaseJpeg(*jpegBuffer);
    *jpegBuffer = nullptr;
    *jpegLength = 0;
    return false;
  }
  return true;
}

void releaseJpeg(uint8_t *jpegBuffer) {
  if (jpegBuffer) free(jpegBuffer);
}
