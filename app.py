import os, time, base64, threading, uuid
from datetime import datetime, timezone

import cv2
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

try:
    from supabase import create_client
except ImportError:
    create_client = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

app = Flask(__name__)
_MODEL_ENV = os.getenv("YOLO_MODEL", os.path.join(BASE_DIR, "best.pt"))
MODEL_PATH = _MODEL_ENV if os.path.isabs(_MODEL_ENV) else os.path.join(BASE_DIR, _MODEL_ENV)
MODEL_UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_models")
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "runtime_models")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_MODEL_UPLOAD_MB", "200")) * 1024 * 1024
CONF = float(os.getenv("YOLO_CONF", "0.35"))
IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
DEVICE_ID = os.getenv("DEVICE_ID", "WEBCAM-01")

# Secrets đọc từ file .env duy nhất (local) hoặc Environment Variables (Render).
# Không commit .env lên GitHub. Biến môi trường hệ thống luôn ưu tiên hơn .env.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_COOLDOWN = int(os.getenv("TELEGRAM_COOLDOWN_SECONDS", "60"))
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://aiumzxboxbbphafebaig.supabase.co").strip()
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "").strip()
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "detection-images").strip()
YOLO_MODEL_BUCKET = os.getenv("YOLO_MODEL_BUCKET", "yolo-models").strip()
YOLO_MODEL_OBJECT = os.getenv("YOLO_MODEL_OBJECT", "").strip()
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "detection_events").strip()

# For this PPE model, send alerts only for these classes. Normal "Deo" is logged
# only when an alert event is generated; adjust ALERT_CLASSES if you want all classes.
ALERT_CLASSES = {"khong deo", "deo sai cach"}
last_alert_at = {}
alert_lock = threading.Lock()
model = None
model_error = None
model_path = MODEL_PATH
model_lock = threading.Lock()
supabase = None
supabase_storage = None
supabase_error = None


def norm(s):
    return str(s).strip().lower().replace("đ", "d")


def init_model():
    global model, model_error, model_path
    source_path = MODEL_PATH
    if YOLO_MODEL_OBJECT:
        if supabase_storage is None:
            model_error = "Không thể tải model từ Supabase: thiếu cấu hình Supabase."
            print("[MODEL]", model_error)
            return
        try:
            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
            source_path = os.path.join(MODEL_CACHE_DIR, os.path.basename(YOLO_MODEL_OBJECT))
            model_bytes = supabase_storage.storage.from_(YOLO_MODEL_BUCKET).download(YOLO_MODEL_OBJECT)
            with open(source_path, "wb") as model_file:
                model_file.write(model_bytes)
            print("[MODEL] Đã tải từ Supabase:", YOLO_MODEL_BUCKET, YOLO_MODEL_OBJECT)
        except Exception as e:
            model_error = f"Không tải được model từ Supabase: {e}"
            print("[MODEL]", model_error)
            return
    if not os.path.isfile(source_path):
        model_error = f"Không tìm thấy model: {source_path}. Hãy đặt best.pt cùng thư mục app.py hoặc cấu hình YOLO_MODEL."
        print("[MODEL]", model_error)
        return
    try:
        model = YOLO(source_path)
        model_path = source_path
        print("[MODEL] Đã tải model:", source_path, "classes:", model.names)
    except Exception as e:
        model_error = str(e)
        print("[MODEL] Lỗi tải model:", e)


def init_services():
    global supabase, supabase_storage, supabase_error
    if create_client is None:
        supabase_error = "Chưa cài supabase package"
    elif SUPABASE_URL and SUPABASE_SECRET_KEY:
        try:
            supabase_storage = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
            supabase = supabase_storage
            supabase.table(SUPABASE_TABLE).select("id").limit(1).execute()
            print("[Supabase] Kết nối OK")
        except Exception as e:
            supabase = None
            supabase_error = str(e)
            print("[Supabase] Lỗi kết nối:", e)
    else:
        supabase_error = "Thiếu SUPABASE_URL hoặc SUPABASE_SECRET_KEY"
    init_model()


def send_telegram(image_bytes, caption):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "Chưa cấu hình Telegram token/chat ID"
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {"photo": ("detection.jpg", image_bytes, "image/jpeg")}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1024]}
        resp = requests.post(url, data=data, files=files, timeout=20)
        resp.raise_for_status()
        body = resp.json()
        if not body.get("ok"):
            raise RuntimeError(str(body))
        return True, "OK"
    except Exception as e:
        print("[Telegram] Gửi lỗi:", e)
        return False, str(e)


def save_supabase(image_bytes, class_name, confidence, telegram_sent):
    if supabase is None:
        return False, None, supabase_error or "Supabase chưa sẵn sàng"
    object_path = f"{datetime.now(timezone.utc):%Y/%m/%d}/{uuid.uuid4().hex}.jpg"
    try:
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            object_path, image_bytes,
            {"content-type": "image/jpeg", "upsert": "false"}
        )
        row = {
            "device_id": DEVICE_ID,
            "class_name": class_name,
            "confidence": float(confidence),
            "image_path": object_path,
            "telegram_sent": bool(telegram_sent),
        }
        supabase.table(SUPABASE_TABLE).insert(row).execute()
        return True, object_path, "OK"
    except Exception as e:
        print("[Supabase] Lưu lỗi:", e)
        return False, object_path, str(e)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/model")
def upload_model():
    global model, model_error, model_path
    uploaded = request.files.get("model")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Hãy chọn một file model .pt"}), 400
    filename = secure_filename(uploaded.filename)
    if not filename.lower().endswith(".pt"):
        return jsonify({"error": "Chỉ chấp nhận file có đuôi .pt"}), 400

    os.makedirs(MODEL_UPLOAD_DIR, exist_ok=True)
    saved_path = os.path.join(MODEL_UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
    try:
        uploaded.save(saved_path)
        new_model = YOLO(saved_path)
        with model_lock:
            model = new_model
            model_path = saved_path
            model_error = None
        print("[MODEL] Đã thay model:", saved_path, "classes:", model.names)
        return jsonify({
            "ok": True,
            "filename": filename,
            "classes": model.names,
            "message": f"Đã tải model {filename} và chuyển sang sử dụng.",
        })
    except Exception as e:
        try:
            os.remove(saved_path)
        except OSError:
            pass
        print("[MODEL] Lỗi upload model:", e)
        return jsonify({"error": f"Không thể tải model: {e}"}), 400


@app.get("/api/health")
def health():
    return jsonify({
        "ok": model is not None,
        "model_loaded": model is not None,
        "model_error": model_error,
        "model_filename": os.path.basename(model_path) if model_path else None,
        "supabase_ready": supabase is not None,
        "supabase_error": supabase_error,
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "classes": model.names if model is not None else {},
    })


@app.post("/api/detect")
def detect():
    with model_lock:
        active_model = model
        active_model_error = model_error
    if active_model is None:
        return jsonify({"error": active_model_error or "Model chưa tải được. Kiểm tra best.pt và log Render."}), 503
    if "frame" not in request.files:
        return jsonify({"error": "Thiếu frame ảnh"}), 400
    raw = request.files["frame"].read()
    if not raw:
        return jsonify({"error": "Ảnh rỗng"}), 400
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Không đọc được ảnh webcam"}), 400

    start = time.perf_counter()
    try:
        result = active_model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        inference_ms = (time.perf_counter() - start) * 1000
        annotated = result.plot()
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            class_name = active_model.names.get(cls_id, str(cls_id)) if isinstance(active_model.names, dict) else active_model.names[cls_id]
            confidence = float(box.conf[0].item())
            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            detections.append({"class_name": str(class_name), "confidence": round(confidence, 4), "bbox": xyxy})
        ok, enc = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return jsonify({"error": "Không mã hóa được ảnh kết quả"}), 500
        out_bytes = enc.tobytes()
        image_b64 = base64.b64encode(out_bytes).decode("ascii")

        # Trigger a notification/storage event only for non-compliant classes,
        # with per-class cooldown to prevent sending a message for every frame.
        alert = next((d for d in detections if norm(d["class_name"]) in ALERT_CLASSES), None)
        telegram_sent = False
        stored = False
        image_path = None
        action_message = ""
        if alert:
            key = norm(alert["class_name"])
            now = time.time()
            with alert_lock:
                allowed = now - last_alert_at.get(key, 0) >= TELEGRAM_COOLDOWN
                if allowed:
                    last_alert_at[key] = now
            if allowed:
                caption = (f"⚠️ CẢNH BÁO NHẬN DIỆN\nThiết bị: {DEVICE_ID}\n"
                           f"Class: {alert['class_name']}\nĐộ tin cậy: {alert['confidence']*100:.1f}%\n"
                           f"Thời gian: {datetime.now().astimezone().strftime('%d/%m/%Y %H:%M:%S')}")
                telegram_sent, tg_msg = send_telegram(out_bytes, caption)
                stored, image_path, db_msg = save_supabase(out_bytes, alert["class_name"], alert["confidence"], telegram_sent)
                action_message = f"Cảnh báo: Telegram {'đã gửi' if telegram_sent else 'chưa gửi'}; Supabase {'đã lưu' if stored else 'chưa lưu'}"
                print("[EVENT]", action_message, "Telegram:", tg_msg, "Supabase:", db_msg)
            else:
                action_message = "Đang trong thời gian chờ trước khi gửi cảnh báo tiếp theo"

        return jsonify({
            "image": "data:image/jpeg;base64," + image_b64,
            "detections": detections,
            "inference_ms": round(inference_ms, 1),
            "server_ms": round((time.perf_counter() - start) * 1000, 1),
            "alert": alert,
            "telegram_sent": telegram_sent,
            "supabase_saved": stored,
            "image_path": image_path,
            "message": action_message,
        })
    except Exception as e:
        print("[DETECT] Lỗi:", e)
        return jsonify({"error": str(e)}), 500


init_services()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False, threaded=True)
