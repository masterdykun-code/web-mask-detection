# YOLO webcam + Telegram + Supabase (Render Free)

Ứng dụng nhận frame webcam từ trình duyệt, chạy model YOLO detection trên server, hiển thị bounding box và gửi cảnh báo qua Telegram. Ảnh cảnh báo cùng metadata được lưu vào Supabase.

## Chạy trên Render Free

Render Free có 512 MB RAM và 0,1 CPU cho mỗi web service. Để giảm RAM, bản deploy này dùng ONNX Runtime và gọi Supabase REST API trực tiếp; không nạp PyTorch/Ultralytics hoặc Supabase SDK vào process chạy web.

1. Xuất file model `best.pt` thành ONNX trên máy có đủ RAM và đã cài Ultralytics:

   ```bash
   pip install ultralytics
   yolo export model=best.pt format=onnx imgsz=320 dynamic=False simplify=False
   ```

   Lệnh tạo `best.onnx` cạnh file `best.pt`. Giữ metadata tên class khi export; không bật `nms=True`.

2. Upload `best.onnx` vào bucket private `yolo-models` trong Supabase. Render tải model này mỗi khi service khởi động.
3. Tạo Blueprint từ `render.yaml`. Thêm các secret `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` và `SUPABASE_SECRET_KEY` trong Render Environment.
4. Mở `/api/health` sau deploy. `model_loaded` phải là `true`; nếu không, xem `model_error` và log deploy.

Web service Free có thể sleep sau 15 phút không có request; lần gọi đầu sau đó cần thời gian khởi động lại. Giới hạn và hành vi hiện tại được mô tả trong [Render Free docs](https://render.com/docs/free) và [Render compute plans](https://render.com/docs/compute-plans).

## Chạy local

1. Tạo môi trường Python 3.11 và cài dependencies:

   ```bash
   python -m venv .venv
   pip install -r requirements.txt
   ```

2. Upload `best.onnx` lên bucket model trong Supabase, rồi thiết lập các biến môi trường bên dưới hoặc trong `.env` cạnh `app.py`.
3. Chạy `python app.py` và mở `http://127.0.0.1:5000`.

## Environment variables

- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`: thông tin bot và chat nhận cảnh báo.
- `SUPABASE_URL`, `SUPABASE_SECRET_KEY`: thông tin Supabase; secret key chỉ đặt ở server.
- `SUPABASE_BUCKET`: bucket ảnh cảnh báo, mặc định `detection-images`.
- `SUPABASE_TABLE`: bảng event, mặc định `detection_events`.
- `YOLO_MODEL_BUCKET`, `YOLO_MODEL_OBJECT`: bucket và object chứa ONNX trong Supabase, mặc định `yolo-models/best.onnx`.
- `YOLO_CLASS_NAMES`: danh sách class phân cách bằng dấu phẩy, chỉ cần đặt nếu ONNX không có metadata tên class. Thứ tự phải khớp với model; hai class cảnh báo mặc định là `Khong deo` và `Deo sai cach`.
- `YOLO_CONF`: ngưỡng confidence, mặc định `0.40`.
- `YOLO_IMGSZ`: kích thước dự phòng nếu ONNX có input động, mặc định `320`.
- `ORT_NUM_THREADS`: số thread inference, mặc định `1`.
- `TELEGRAM_COOLDOWN_SECONDS`: khoảng chờ toàn hệ thống giữa các cảnh báo, mặc định `20` giây.
- `DEVICE_ID`: tên thiết bị trong event, mặc định `WEBCAM-01`.
- `MAX_FRAME_BYTES`: giới hạn kích thước request, mặc định `2097152` byte.

## Supabase schema

Bảng `public.detection_events` cần các cột `device_id` (text), `class_name` (text), `confidence` (real), `image_path` (text), `telegram_sent` (boolean). Các cột `id`, `detected_at`, `created_at` có thể dùng giá trị mặc định. Tạo hai bucket `yolo-models` và `detection-images`; bucket model nên là private. Server dùng `SUPABASE_SECRET_KEY` để tải model, upload ảnh và ghi event.

## Luồng hoạt động

- Trình duyệt gửi JPEG tối đa 640 px; mỗi frame cách nhau 350 ms sau khi request trước hoàn tất, không gửi chồng request.
- Inference giữ input ONNX 320 px; Telegram và Supabase chạy nền để không chặn response nhận diện.
- Giao diện đo ba độ trễ: `Suy luận cloud` là thời gian tiền xử lý + ONNX + hậu xử lý trong `predict()` trên Render; `Local ↔ Cloud` là thời gian từ lúc gửi request đến khi nhận và phân tích xong JSON; `Chụp frame → bbox` là thời gian trên trình duyệt từ lúc bắt đầu lấy frame đến khi vẽ bbox. Mỗi số lớn là lần gần nhất; dòng nhỏ là trung bình trượt tối đa 20 frame thành công.
- API trả `server_ms` để đo thời gian xử lý trong route trên server, gồm đọc/giải mã ảnh, suy luận và chuẩn bị cảnh báo; không gồm truyền request/response. Giá trị này phục vụ phân tích server, còn `inference_ms` chỉ đo pipeline nhận diện.
- Server tải model ONNX từ Supabase theo từng lần khởi động và chỉ giữ một ONNX Runtime session trong RAM.
- Chỉ class `Khong deo` và `Deo sai cach` tạo event Telegram/Supabase; các class khác vẫn hiện trên giao diện.
- Render Free dùng filesystem tạm thời; file tải về chỉ là cache và model sẽ được tải lại sau khi service khởi động lại.

## Hai chế độ camera

- **Camera laptop:** trình duyệt chụp frame và gửi trực tiếp đến `/api/detect` trên Render.
- **Camera IoT:** ESP32-CAM gửi JPEG trực tiếp bằng HTTPS đến `/api/iot/frame`; Render chạy cùng model ONNX. Trình duyệt lấy frame và bbox mới nhất từ `/api/iot/latest`.
- ESP32 không gửi vào IP `192.168.137.1` và không dùng giao thức TCP 4 byte + JPEG nữa. IP đó chỉ thuộc mạng riêng của laptop. Hãy nạp firmware trong `firmware/esp32_cam` để ESP32 gọi domain HTTPS của Render.
- ESP32 đo thời gian upload đến khi nhận phản hồi Render rồi gửi số đo qua `/api/iot/metrics`. Độ trễ toàn trình dùng thời gian chụp đã đồng bộ NTP trên ESP32 đến lúc trình duyệt vẽ bbox.
- Render chỉ giữ frame IoT mới nhất trong RAM; frame và sequence sẽ mất khi service khởi động lại. Mỗi request IoT vẫn chạy cảnh báo/lưu event như camera laptop.

## Cấu hình bảo vệ camera IoT

Trong Render Dashboard → service → **Environment**, đặt hai secret khác nhau:

- `IOT_DEVICE_TOKEN`: khóa gửi frame, điền vào `firmware/esp32_cam/secrets.h`.
- `IOT_VIEW_TOKEN`: khóa xem frame, nhập trong giao diện khi chọn chế độ IoT.

`render.yaml` khai báo hai biến này là secret không đồng bộ. Không commit `secrets.h` hoặc các giá trị token.
`/api/health` có `iot_upload_configured` và `iot_view_configured` để kiểm tra hai khóa đã được Render nạp.

## Firmware ESP32-CAM

Mã nguồn được chia trong [`firmware/esp32_cam`](firmware/esp32_cam): `esp32_cam.ino` quản lý vòng lặp, `camera_capture.*` cấu hình và chụp GC2145, `cloud_sender.*` gửi HTTPS và báo thời gian round-trip, `config.h` chứa domain và thông số, `secrets.example.h` là mẫu thông tin Wi-Fi/token. Chi tiết nạp firmware và thay URL nằm trong [hướng dẫn firmware](firmware/esp32_cam/README.md).
