#include <Arduino.h>
#include <WiFi.h>
#include <time.h>

#include "camera_capture.h"
#include "cloud_sender.h"
#include "config.h"

static void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print('.');
  }
  Serial.println();
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

static void synchronizeClock() {
  configTime(0, 0, "pool.ntp.org", "time.google.com");
  const uint32_t startedAt = millis();
  while (time(nullptr) < 1700000000 && millis() - startedAt < 20000) {
    delay(500);
  }
  if (time(nullptr) >= 1700000000) Serial.println("NTP clock: synchronized");
  else Serial.println("NTP clock: unavailable; end-to-end timing will use browser fallback");
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  Serial.println();
  Serial.println("ESP32-CAM GC2145 -> Render HTTPS");

  if (psramFound()) {
    Serial.printf("PSRAM: %u KB\n", ESP.getPsramSize() / 1024);
  } else {
    Serial.println("PSRAM not found");
  }

  if (!initializeCamera()) {
    while (true) delay(1000);
  }
  Serial.println("Camera: OK");

  connectWifi();
  synchronizeClock();
  Serial.print("Render endpoint: ");
  Serial.print(RENDER_BASE_URL);
  Serial.println("/api/iot/frame");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected; reconnecting...");
    WiFi.disconnect();
    connectWifi();
    synchronizeClock();
  }

  uint8_t *jpegBuffer = nullptr;
  size_t jpegLength = 0;
  int64_t capturedAtMs = 0;
  if (!captureJpeg(&jpegBuffer, &jpegLength, &capturedAtMs)) {
    delay(100);
    return;
  }

  UploadResult result{};
  const bool uploaded = uploadJpegToRender(jpegBuffer, jpegLength, capturedAtMs, &result);
  releaseJpeg(jpegBuffer);

  if (uploaded) {
    Serial.printf("Frame %u uploaded; cloud round-trip: %u ms\n", result.sequence, result.roundTripMs);
    if (!reportUploadRoundTrip(result.sequence, result.roundTripMs)) {
      Serial.println("Could not report round-trip metric");
    }
  } else {
    Serial.printf("Render upload failed; HTTP status: %d; body: %s\n",
                  result.httpStatus, result.responseBody.c_str());
  }

  delay(UPLOAD_PAUSE_MS);
}
