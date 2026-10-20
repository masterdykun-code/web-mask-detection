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

- Trình duyệt gửi JPEG tối đa 960 px mỗi 1,2 giây; server chỉ xử lý tuần tự.
- Server tải model ONNX từ Supabase theo từng lần khởi động và chỉ giữ một ONNX Runtime session trong RAM.
- Chỉ class `Khong deo` và `Deo sai cach` tạo event Telegram/Supabase; các class khác vẫn hiện trên giao diện.
- Render Free dùng filesystem tạm thời; file tải về chỉ là cache và model sẽ được tải lại sau khi service khởi động lại.
