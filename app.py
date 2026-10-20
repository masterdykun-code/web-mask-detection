import base64
from concurrent.futures import ThreadPoolExecutor
import hmac
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import quote

# Keep numerical libraries from creating extra worker threads on small instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request

from detector import OnnxDetector

cv2.setNumThreads(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "runtime_models")

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_FRAME_BYTES", str(2 * 1024 * 1024)))

CONF = float(os.getenv("YOLO_CONF", "0.40"))
IMGSZ = int(os.getenv("YOLO_IMGSZ", "320"))
DEVICE_ID = os.getenv("DEVICE_ID", "WEBCAM-01")
ORT_NUM_THREADS = max(1, int(os.getenv("ORT_NUM_THREADS", "1")))

# Secrets are read from .env locally or from Render environment variables.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
ALERT_COOLDOWN = int(os.getenv("TELEGRAM_COOLDOWN_SECONDS", "20"))
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://aiumzxboxbbphafebaig.supabase.co").strip().rstrip("/")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "").strip()
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "detection-images").strip()
YOLO_MODEL_BUCKET = os.getenv("YOLO_MODEL_BUCKET", "yolo-models").strip()
YOLO_MODEL_OBJECT = os.getenv("YOLO_MODEL_OBJECT", "best.onnx").strip()
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "detection_events").strip()
YOLO_CLASS_NAMES = [name.strip() for name in os.getenv("YOLO_CLASS_NAMES", "").split(",") if name.strip()]
IOT_DEVICE_TOKEN = os.getenv("IOT_DEVICE_TOKEN", "").strip()
IOT_VIEW_TOKEN = os.getenv("IOT_VIEW_TOKEN", "").strip()
IOT_INSTANCE_ID = uuid.uuid4().hex

ALERT_CLASSES = {"khong deo", "deo sai cach"}
last_alert_at = 0.0
alert_lock = threading.Lock()
alert_status_lock = threading.Lock()
alert_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="alert-delivery")
latest_alert_status = {"event_id": None, "state": "idle", "message": ""}
model = None
model_error = None
model_path = None
supabase = None
supabase_error = None
iot_result_lock = threading.Lock()
latest_iot_result = {
    "instance_id": IOT_INSTANCE_ID,
    "sequence": 0,
    "frame_jpeg": None,
    "width": 0,
    "height": 0,
    "detections": [],
    "inference_ms": None,
    "server_ms": None,
    "captured_at_ms": None,
    "device_rtt_ms": None,
    "metrics_sequence": 0,
    "alert": None,
    "message": "",
    "alert_status": {},
}


@app.errorhandler(413)
def request_too_large(error):
    return jsonify({"error": "Frame quá lớn; giới hạn hiện tại là 2 MB."}), 413


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Lỗi server khi xử lý frame"}), 500


def norm(value):
    return str(value).strip().lower().replace("đ", "d")


def supabase_headers(extra=None):
    headers = {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
    }
    if extra:
        headers.update(extra)
    return headers


def storage_object_url(bucket, object_path):
    return (
        f"{SUPABASE_URL}/storage/v1/object/"
        f"{quote(bucket, safe='')}/{quote(object_path, safe='/')}"
    )


def init_model():
    global model, model_error, model_path
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        model_error = "Thiếu SUPABASE_URL hoặc SUPABASE_SECRET_KEY để tải model."
        print("[MODEL]", model_error)
        return
    if not YOLO_MODEL_OBJECT.lower().endswith(".onnx"):
        model_error = "YOLO_MODEL_OBJECT phải trỏ tới file ONNX, ví dụ best.onnx."
        print("[MODEL]", model_error)
        return

    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        source_path = os.path.join(MODEL_CACHE_DIR, os.path.basename(YOLO_MODEL_OBJECT))
        download_path = source_path + ".download"
        url = storage_object_url(YOLO_MODEL_BUCKET, YOLO_MODEL_OBJECT)
        downloaded = 0
        try:
            with requests.get(
                url,
                headers=supabase_headers(),
                stream=True,
                timeout=(10, 120),
            ) as response:
                response.raise_for_status()
                with open(download_path, "wb") as model_file:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            model_file.write(chunk)
                            downloaded += len(chunk)
            if downloaded == 0:
                raise RuntimeError("File ONNX tải về từ Supabase bị rỗng.")
            os.replace(download_path, source_path)
        finally:
            if os.path.exists(download_path):
                os.remove(download_path)

        loaded_model = OnnxDetector(
            source_path, imgsz=IMGSZ, num_threads=ORT_NUM_THREADS, class_names=YOLO_CLASS_NAMES
        )
        model = loaded_model
        model_path = source_path
        model_error = None
        print(
            "[MODEL] Đã tải ONNX từ Supabase:",
            f"{YOLO_MODEL_BUCKET}/{YOLO_MODEL_OBJECT}",
            "classes:",
            model.names,
        )
    except Exception as error:
        model = None
        model_error = str(error)
        print("[MODEL] Lỗi tải model:", error)


def init_services():
    global supabase, supabase_error
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        supabase_error = "Thiếu SUPABASE_URL hoặc SUPABASE_SECRET_KEY"
    else:
        try:
            url = f"{SUPABASE_URL}/rest/v1/{quote(SUPABASE_TABLE, safe='')}?select=id&limit=1"
            response = requests.get(url, headers=supabase_headers(), timeout=(5, 15))
            response.raise_for_status()
            supabase = True
            supabase_error = None
            print("[Supabase] Kết nối OK")
        except Exception as error:
            supabase = None
            supabase_error = str(error)
            print("[Supabase] Lỗi kết nối:", error)
    init_model()


def send_telegram(image_bytes, caption):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "Chưa cấu hình Telegram token/chat ID"
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {"photo": ("detection.jpg", image_bytes, "image/jpeg")}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1024]}
        response = requests.post(url, data=data, files=files, timeout=20)
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            raise RuntimeError(str(body))
        return True, "OK"
    except Exception as error:
        print("[Telegram] Gửi lỗi:", error)
        return False, str(error)


def save_supabase(image_bytes, class_name, confidence, telegram_sent):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return False, None, supabase_error or "Supabase chưa sẵn sàng"
    object_path = f"{datetime.now(timezone.utc):%Y/%m/%d}/{uuid.uuid4().hex}.jpg"
    try:
        upload_response = requests.post(
            storage_object_url(SUPABASE_BUCKET, object_path),
            headers=supabase_headers({"Content-Type": "image/jpeg", "x-upsert": "false"}),
            data=image_bytes,
            timeout=(10, 30),
        )
        upload_response.raise_for_status()
        row = {
            "device_id": DEVICE_ID,
            "class_name": class_name,
            "confidence": float(confidence),
            "image_path": object_path,
            "telegram_sent": bool(telegram_sent),
        }
        table_url = f"{SUPABASE_URL}/rest/v1/{quote(SUPABASE_TABLE, safe='')}"
        insert_response = requests.post(
            table_url,
            headers=supabase_headers({"Content-Type": "application/json", "Prefer": "return=minimal"}),
            json=row,
            timeout=(10, 30),
        )
        insert_response.raise_for_status()
        return True, object_path, "OK"
    except Exception as error:
        print("[Supabase] Lưu lỗi:", error)
        return False, object_path, str(error)


def set_alert_status(event_id, state, message):
    with alert_status_lock:
        if state == "queued" or latest_alert_status["event_id"] == event_id:
            latest_alert_status.update({"state": state, "message": message})
            latest_alert_status["event_id"] = event_id
        return latest_alert_status.copy()


def get_alert_status():
    with alert_status_lock:
        return latest_alert_status.copy()


def deliver_alert(event_id, image_bytes, caption, class_name, confidence):
    set_alert_status(event_id, "sending", "Đang gửi cảnh báo Telegram và lưu ảnh lên Supabase…")
    try:
        telegram_sent, telegram_message = send_telegram(image_bytes, caption)
        stored, image_path, database_message = save_supabase(
            image_bytes, class_name, confidence, telegram_sent
        )
    except Exception as error:
        set_alert_status(event_id, "failed", "Gửi cảnh báo hoặc lưu ảnh gặp lỗi.")
        print("[EVENT] Lỗi xử lý cảnh báo nền:", error)
        return
    action_message = (
        f"Cảnh báo: Telegram {'đã gửi' if telegram_sent else 'chưa gửi'}; "
        f"Supabase {'đã lưu' if stored else 'chưa lưu'}"
    )
    state = "done" if telegram_sent and stored else "partial"
    set_alert_status(event_id, state, action_message)
    print(
        "[EVENT]",
        action_message,
        "Telegram:",
        telegram_message,
        "Supabase:",
        database_message,
        "Image:",
        image_path,
    )


def draw_detections(frame, detections):
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
        color = (113, 113, 251) if norm(detection["class_name"]) in ALERT_CLASSES else (74, 222, 128)
        label = f"{detection['class_name']} {detection['confidence'] * 100:.1f}%"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_top = max(text_height + baseline, y1)
        cv2.rectangle(annotated, (x1, label_top - text_height - baseline), (x1 + text_width + 8, label_top), color, -1)
        cv2.putText(annotated, label, (x1 + 4, label_top - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (7, 17, 31), 2)
    return annotated


@app.get("/")
def index():
    return render_template("index.html")


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
        "iot_upload_configured": bool(IOT_DEVICE_TOKEN),
        "iot_view_configured": bool(IOT_VIEW_TOKEN),
        "classes": model.names if model is not None else {},
    })


@app.post("/api/detect")
def detect():
    server_start = time.perf_counter()
    active_model = model
    active_model_error = model_error
    if active_model is None:
        return jsonify({"error": active_model_error or "Model chưa tải được. Kiểm tra best.onnx và log Render."}), 503
    if "frame" not in request.files:
        return jsonify({"error": "Thiếu frame ảnh"}), 400
    raw = request.files["frame"].read()
    if not raw:
        return jsonify({"error": "Ảnh rỗng"}), 400
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Không đọc được ảnh webcam"}), 400

    result, status = process_detection(frame, server_start)
    return jsonify(result), status


def process_detection(frame, server_start):
    global last_alert_at
    active_model = model
    if active_model is None:
        return {"error": model_error or "Model chưa tải được."}, 503
    try:
        inference_start = time.perf_counter()
        detections = active_model.predict(frame, CONF)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        alert = next((item for item in detections if norm(item["class_name"]) in ALERT_CLASSES), None)
        telegram_sent = False
        stored = False
        image_path = None
        action_message = ""

        if alert:
            now = time.time()
            with alert_lock:
                allowed = now - last_alert_at >= ALERT_COOLDOWN
                if allowed:
                    last_alert_at = now
            if allowed:
                annotated = draw_detections(frame, detections)
                ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if not ok:
                    return {"error": "Không mã hóa được ảnh cảnh báo"}, 500
                image_bytes = encoded.tobytes()
                caption = (
                    f"⚠️ CẢNH BÁO NHẬN DIỆN\nThiết bị: {DEVICE_ID}\n"
                    f"Class: {alert['class_name']}\nĐộ tin cậy: {alert['confidence'] * 100:.1f}%\n"
                    f"Thời gian: {datetime.now().astimezone().strftime('%d/%m/%Y %H:%M:%S')}"
                )
                event_id = uuid.uuid4().hex
                action_message = "Đã phát hiện cảnh báo; đang gửi Telegram và lưu ảnh nền."
                telegram_sent = None
                stored = None
                set_alert_status(event_id, "queued", action_message)
                alert_executor.submit(
                    deliver_alert,
                    event_id,
                    image_bytes,
                    caption,
                    alert["class_name"],
                    alert["confidence"],
                )
            else:
                action_message = "Đang trong thời gian chờ trước khi gửi cảnh báo tiếp theo"

        return {
            "detections": detections,
            "inference_ms": round(inference_ms, 1),
            "server_ms": round((time.perf_counter() - server_start) * 1000, 1),
            "alert": alert,
            "telegram_sent": telegram_sent,
            "supabase_saved": stored,
            "image_path": image_path,
            "message": action_message,
            "alert_status": get_alert_status(),
        }, 200
    except Exception as error:
        print("[DETECT] Lỗi:", error)
        return {"error": str(error)}, 500


def token_matches(expected, supplied):
    return bool(expected and supplied and hmac.compare_digest(expected, supplied))


@app.post("/api/iot/frame")
def iot_frame():
    global latest_iot_result
    server_start = time.perf_counter()
    if not IOT_DEVICE_TOKEN:
        return jsonify({"error": "IOT_DEVICE_TOKEN chưa được cấu hình trên Render."}), 503
    if not token_matches(IOT_DEVICE_TOKEN, request.headers.get("X-IoT-Device-Token", "")):
        return jsonify({"error": "Mã thiết bị IoT không hợp lệ."}), 401
    if model is None:
        return jsonify({"error": model_error or "Model chưa tải được."}), 503

    raw = request.get_data(cache=False)
    if not raw:
        return jsonify({"error": "Ảnh JPEG rỗng."}), 400
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Không giải mã được ảnh JPEG từ ESP32-CAM."}), 400

    result, status = process_detection(frame, server_start)
    if status != 200:
        return jsonify(result), status

    try:
        captured_at_ms = int(request.headers.get("X-IoT-Captured-At-Ms", "0"))
    except (TypeError, ValueError):
        captured_at_ms = 0
    height, width = frame.shape[:2]
    with iot_result_lock:
        sequence = latest_iot_result["sequence"] + 1
        latest_iot_result = {
            "instance_id": IOT_INSTANCE_ID,
            "sequence": sequence,
            "frame_jpeg": base64.b64encode(raw).decode("ascii"),
            "width": width,
            "height": height,
            "detections": result["detections"],
            "inference_ms": result["inference_ms"],
            "server_ms": result["server_ms"],
            "captured_at_ms": captured_at_ms if captured_at_ms > 0 else None,
            "device_rtt_ms": None,
            "metrics_sequence": 0,
            "alert": result["alert"],
            "message": result["message"],
            "alert_status": result["alert_status"],
        }

    response = jsonify({"ok": True, "sequence": sequence, "inference_ms": result["inference_ms"]})
    response.headers["X-IoT-Sequence"] = str(sequence)
    return response


@app.post("/api/iot/metrics")
def iot_metrics():
    global latest_iot_result
    if not IOT_DEVICE_TOKEN:
        return jsonify({"error": "IOT_DEVICE_TOKEN chưa được cấu hình trên Render."}), 503
    if not token_matches(IOT_DEVICE_TOKEN, request.headers.get("X-IoT-Device-Token", "")):
        return jsonify({"error": "Mã thiết bị IoT không hợp lệ."}), 401
    data = request.get_json(silent=True) or {}
    try:
        sequence = int(data.get("sequence", 0))
        device_rtt_ms = float(data.get("device_rtt_ms"))
    except (TypeError, ValueError):
        return jsonify({"error": "Thiếu sequence hoặc device_rtt_ms không hợp lệ."}), 400
    if device_rtt_ms < 0:
        return jsonify({"error": "device_rtt_ms không hợp lệ."}), 400
    with iot_result_lock:
        if latest_iot_result["sequence"] == sequence:
            latest_iot_result["device_rtt_ms"] = round(device_rtt_ms, 1)
            latest_iot_result["metrics_sequence"] += 1
    return jsonify({"ok": True})


@app.get("/api/iot/latest")
def iot_latest():
    if not IOT_VIEW_TOKEN:
        return jsonify({"error": "IOT_VIEW_TOKEN chưa được cấu hình trên Render."}), 503
    if not token_matches(IOT_VIEW_TOKEN, request.headers.get("X-IoT-View-Token", "")):
        return jsonify({"error": "Mã xem camera IoT không hợp lệ."}), 401
    try:
        after = int(request.args.get("after", "0"))
    except ValueError:
        return jsonify({"error": "Tham số after không hợp lệ."}), 400
    after_instance = request.args.get("instance", "")
    with iot_result_lock:
        snapshot = latest_iot_result.copy()
    if snapshot["sequence"] == 0 or (
        snapshot["sequence"] <= after and after_instance == snapshot["instance_id"]
    ):
        return jsonify({
            "updated": False,
            "instance_id": snapshot["instance_id"],
            "sequence": snapshot["sequence"],
            "device_rtt_ms": snapshot["device_rtt_ms"],
            "metrics_sequence": snapshot["metrics_sequence"],
        })
    snapshot["updated"] = True
    return jsonify(snapshot)


init_services()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False, threaded=False)
