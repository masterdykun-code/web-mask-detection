# YOLO webcam + Telegram + Supabase (Render)

## Tính năng

- Webcam trình duyệt gửi frame lên Flask để suy luận bằng model YOLO đang được chọn.
- Model luôn được tải từ file `best.pt` trong bucket Supabase private khi server khởi động.
- Trả ảnh có bounding box về website, kèm class/confidence và latency.
- Khi phát hiện `Khong deo` hoặc `Deo sai cach`, gửi ảnh cảnh báo lên Telegram và upload ảnh đã vẽ bbox lên bucket Supabase `detection-images`, sau đó ghi metadata vào bảng `detection_events`.
- Mỗi class có cooldown, mặc định 60 giây để hạn chế spam.

## Chạy local

1. Đặt `best.pt` cạnh `app.py`.
2. Python 3.11 khuyến nghị (YOLO/PyTorch trên Render tương thích tốt hơn Python 3.13).
3. `python -m venv .venv` rồi kích hoạt môi trường.
4. `pip install -r requirements.txt`
5. Thiết lập biến môi trường như danh sách dưới, sau đó `python app.py`.
6. Mở `http://127.0.0.1:5000`.

## Environment variables

- `TELEGRAM_BOT_TOKEN`: token bot từ BotFather (server-side secret).
- `TELEGRAM_CHAT_ID`: chat ID người nhận. Mở bot, nhấn `/start` trước.
- `SUPABASE_URL`: `https://aiumzxboxbbphafebaig.supabase.co`
- `SUPABASE_SECRET_KEY`: secret key của Supabase, chỉ lưu ở server/Render Environment. Không commit lên GitHub, không đặt trong JavaScript.
- `SUPABASE_BUCKET`: `detection-images` (bucket private).
- `SUPABASE_TABLE`: `detection_events`.
- `YOLO_MODEL_BUCKET`: bucket Supabase private chứa model, mặc định `yolo-models`.
- `YOLO_MODEL_OBJECT`: đường dẫn file model trong bucket model, mặc định `best.pt`.
- `TELEGRAM_COOLDOWN_SECONDS`: thời gian tối thiểu giữa hai lần gửi cảnh báo, mặc định `20` giây.
- `YOLO_CONF`: ngưỡng confidence, mặc định `0.40`.
- `YOLO_IMGSZ`: kích thước ảnh inference, mặc định `320` để giảm RAM trên Render Free.
- `MAX_MODEL_UPLOAD_MB`: kích thước file `.pt` upload tối đa, mặc định `200` MB.

## Deploy Render

1. Upload `best.pt` vào bucket private `yolo-models`. Trên Render, cấu hình `YOLO_MODEL_BUCKET=yolo-models` và `YOLO_MODEL_OBJECT=best.pt` để app tải model từ Supabase lúc khởi động. Bucket `SUPABASE_BUCKET=detection-images` vẫn dùng cho ảnh cảnh báo.
2. Push project lên GitHub.
3. Render > New > Blueprint và chọn repository để đọc `render.yaml`, hoặc tạo Web Service thủ công.
4. Trong Environment, nhập `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `SUPABASE_SECRET_KEY`. Không cần thêm dấu ngoặc kép.
5. Dùng Build Command `pip install -r requirements.txt`; Start Command `gunicorn app:app --workers 1 --threads 2 --timeout 180`.
6. Render cần đủ RAM để tải PyTorch/YOLO; gói Free có thể thiếu RAM hoặc bị sleep. Kiểm tra chi phí instance trước khi chọn.

## Supabase schema yêu cầu

Bảng `public.detection_events` cần các cột `device_id` (text), `class_name` (text), `confidence` (real), `image_path` (text), `telegram_sent` (boolean); các cột `id`, `detected_at`, `created_at` có thể tự sinh default như đã tạo. Bucket `detection-images` cần tồn tại. Dùng secret key server-side để upload bucket private và insert dữ liệu; không dùng publishable key cho các thao tác này.

## Lưu ý

- Mặc định chỉ gửi Telegram và ghi event vào Supabase với class `Khong deo` / `Deo sai cach`; class bình thường `Deo` không gửi cảnh báo. Mỗi cảnh báo chỉ được gửi tối đa một lần trong 20 giây trên toàn hệ thống.
- Bucket private: `image_path` lưu đường dẫn object, không phải URL public.
- Cấu hình Telegram bằng cách gửi `/start` cho bot và lấy đúng chat ID người dùng/group, không dùng bot ID.
