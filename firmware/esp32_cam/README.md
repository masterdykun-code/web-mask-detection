# ESP32-CAM GC2145 → Render

Firmware này gửi JPEG trực tiếp từ ESP32-CAM lên Render bằng HTTPS. Nó thay giao thức TCP raw cũ (4 byte độ dài + JPEG) bằng HTTP POST.

## 1. Cấu hình Render

Trong Render Dashboard → web service → **Environment**, đặt:

- `IOT_DEVICE_TOKEN`: chuỗi bí mật dùng cho ESP32 gửi ảnh.
- `IOT_VIEW_TOKEN`: chuỗi bí mật để mở luồng trên giao diện.

Đặt hai giá trị khác nhau. Sau khi lưu, đợi Render deploy lại.

## 2. Cấu hình Arduino

1. Mở thư mục `firmware/esp32_cam` như một sketch Arduino. Sao chép file `board_config.h` đang dùng cho board M21-45 vào cùng thư mục; file này phải khai báo các macro `Y2_GPIO_NUM` đến `Y9_GPIO_NUM`, `XCLK_GPIO_NUM`, `PCLK_GPIO_NUM`, `VSYNC_GPIO_NUM`, `HREF_GPIO_NUM`, `SIOD_GPIO_NUM`, `SIOC_GPIO_NUM`, `PWDN_GPIO_NUM` và `RESET_GPIO_NUM`.
2. Sao chép `secrets.example.h` thành `secrets.h`; điền SSID, mật khẩu Wi-Fi và đúng `IOT_DEVICE_TOKEN` đã đặt trên Render. `secrets.h` được `.gitignore` bỏ qua.
3. Trong `config.h`, thay domain ở `RENDER_BASE_URL` bằng domain HTTPS hiển thị trong Render Dashboard → service → **Settings** → **Domains**. Ví dụ:

   ```cpp
   static constexpr char RENDER_BASE_URL[] = "https://render-yolo-web.onrender.com";
   ```

   Không thêm `/api/iot/frame` vào domain; firmware tự nối endpoint này.
4. Chọn đúng board ESP32-CAM và nạp `esp32_cam.ino`. Mở Serial Monitor ở 115200 baud.

Thay khai báo cũ:

```cpp
const char *SERVER_IP = "192.168.137.1";
const uint16_t SERVER_PORT = 5002;
```

bằng domain HTTPS trong `config.h` (`RENDER_BASE_URL`). Không dùng IP LAN hoặc cổng 5002 nữa. Nếu Render cấp domain khác, dùng đúng domain trong Dashboard.

Khi chạy, Serial Monitor báo mã HTTP, sequence frame và round-trip ESP32 ↔ Render. Frame VGA và JPEG quality 65 được gửi đồng bộ; ESP32 đợi phản hồi trước khi chụp lượt tiếp để hạn chế tải Render Free.

## 3. Xem camera IoT

Mở web app Render, chọn **Camera IoT · ESP32-CAM**, nhập `IOT_VIEW_TOKEN`, rồi bấm **Kết nối IoT**. Giao diện polling frame mới nhất và vẽ bbox theo đúng kích thước frame GC2145.

## Ghi chú TLS

Firmware dùng HTTPS nhưng hiện gọi `WiFiClientSecure::setInsecure()`, nên dữ liệu được mã hóa nhưng ESP32 không xác thực chứng thư máy chủ. Dùng mạng tin cậy; có thể chuyển sang `setCACert()` nếu muốn bật xác thực CA.
