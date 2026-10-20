#include "cloud_sender.h"

#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <stdio.h>
#include <stdlib.h>

#include "config.h"

namespace {
bool openHttpsClient(WiFiClientSecure &tls, HTTPClient &http, const char *url) {
  // TLS encrypts traffic; certificate validation can be enabled with Render's CA certificate.
  tls.setInsecure();
  tls.setTimeout(HTTP_TIMEOUT_MS);
  http.setTimeout(HTTP_TIMEOUT_MS);
  return http.begin(tls, url);
}
}

bool uploadJpegToRender(
    const uint8_t *jpegBuffer,
    size_t jpegLength,
    int64_t capturedAtMs,
    UploadResult *result) {
  if (!jpegBuffer || jpegLength == 0 || !result) return false;
  result->httpStatus = -1;
  result->roundTripMs = 0;
  result->sequence = 0;
  result->responseBody = "";

  WiFiClientSecure tls;
  HTTPClient http;
  const String url = String(RENDER_BASE_URL) + "/api/iot/frame";
  if (!openHttpsClient(tls, http, url.c_str())) {
    Serial.println("Could not start HTTPS request to Render");
    return false;
  }

  const char *responseHeaders[] = {"X-IoT-Sequence"};
  http.collectHeaders(responseHeaders, 1);
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-IoT-Device-Token", IOT_DEVICE_TOKEN);
  char captureTimeHeader[24];
  snprintf(captureTimeHeader, sizeof(captureTimeHeader), "%lld", static_cast<long long>(capturedAtMs));
  http.addHeader("X-IoT-Captured-At-Ms", captureTimeHeader);

  const uint32_t startedAt = millis();
  result->httpStatus = http.POST(const_cast<uint8_t *>(jpegBuffer), jpegLength);
  if (result->httpStatus > 0) {
    result->responseBody = http.getString();
    const String sequenceHeader = http.header("X-IoT-Sequence");
    if (sequenceHeader.length()) result->sequence = strtoul(sequenceHeader.c_str(), nullptr, 10);
    if (result->sequence == 0) {
      const int keyPosition = result->responseBody.indexOf("\"sequence\":");
      if (keyPosition >= 0) {
        const char *sequenceStart = result->responseBody.c_str() + keyPosition + 11;
        result->sequence = strtoul(sequenceStart, nullptr, 10);
      }
    }
  }
  result->roundTripMs = millis() - startedAt;
  http.end();
  return result->httpStatus >= 200 && result->httpStatus < 300 && result->sequence > 0;
}

bool reportUploadRoundTrip(uint32_t sequence, uint32_t roundTripMs) {
  if (sequence == 0) return false;
  WiFiClientSecure tls;
  HTTPClient http;
  const String url = String(RENDER_BASE_URL) + "/api/iot/metrics";
  if (!openHttpsClient(tls, http, url.c_str())) {
    return false;
  }
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-IoT-Device-Token", IOT_DEVICE_TOKEN);
  const String body = String("{\"sequence\":") + sequence +
      String(",\"device_rtt_ms\":") + roundTripMs + "}";
  const int status = http.POST(body);
  http.end();
  return status >= 200 && status < 300;
}
