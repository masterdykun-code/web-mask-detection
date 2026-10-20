"""ONNX Runtime inference for Ultralytics detection exports."""

import ast
import json

import cv2
import numpy as np
import onnxruntime as ort


class OnnxDetector:
    """Small CPU-only detector for Ultralytics detection ONNX exports."""

    def __init__(self, path, imgsz=320, num_threads=1, class_names=None):
        self.imgsz = int(imgsz)
        self.num_threads = max(1, int(num_threads))
        self.class_names = [str(name) for name in (class_names or [])]
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.num_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = model_input.shape
        self.input_height = shape[2] if isinstance(shape[2], int) else self.imgsz
        self.input_width = shape[3] if isinstance(shape[3], int) else self.imgsz
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.end2end = metadata.get("end2end", "").lower() == "true"
        self.names = self._read_names()
        if not self.names:
            raise RuntimeError(
                "ONNX không có metadata tên class; hãy export lại bằng Ultralytics "
                "hoặc đặt YOLO_CLASS_NAMES, ví dụ: Khong deo,Deo sai cach,Deo."
            )

    def _read_names(self):
        raw = self.session.get_modelmeta().custom_metadata_map.get("names")
        names = None
        if raw:
            try:
                names = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                try:
                    names = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    names = None
        if isinstance(names, dict):
            try:
                return {int(key): str(value) for key, value in names.items()}
            except (TypeError, ValueError):
                return None
        if isinstance(names, (list, tuple)):
            return {index: str(value) for index, value in enumerate(names)}
        if self.class_names:
            return {index: name for index, name in enumerate(self.class_names)}
        return None

    def predict(self, frame, confidence_threshold):
        frame_height, frame_width = frame.shape[:2]
        input_width, input_height = int(self.input_width), int(self.input_height)
        scale = min(input_width / frame_width, input_height / frame_height)
        resized_width = max(1, int(round(frame_width * scale)))
        resized_height = max(1, int(round(frame_height * scale)))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_left = (input_width - resized_width) // 2
        pad_top = (input_height - resized_height) // 2
        padded = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        padded[pad_top:pad_top + resized_height, pad_left:pad_left + resized_width] = resized
        tensor = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1.0 / 255.0,
            size=(input_width, input_height),
            swapRB=True,
            crop=False,
            ddepth=cv2.CV_32F,
        )

        output = self.session.run(None, {self.input_name: tensor})[0]
        predictions = np.asarray(output)
        while predictions.ndim > 2 and predictions.shape[0] == 1:
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise RuntimeError(f"Định dạng output ONNX không được hỗ trợ: {predictions.shape}")

        # Ultralytics end-to-end exports return [x1, y1, x2, y2, confidence,
        # class_id]. Non-end-to-end exports return center-xywh plus class scores.
        if self.end2end:
            if predictions.shape[1] != 6:
                raise RuntimeError(f"Output end-to-end ONNX không đúng định dạng [N, 6]: {predictions.shape}")
            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            class_ids = np.rint(predictions[:, 5]).astype(np.int32)
            valid = (scores >= confidence_threshold) & (class_ids >= 0) & (class_ids < len(self.names))
            boxes = boxes[valid]
            scores = scores[valid]
            class_ids = class_ids[valid]
            if not len(scores):
                return []
            x1 = np.clip((boxes[:, 0] - pad_left) / scale, 0, frame_width - 1)
            y1 = np.clip((boxes[:, 1] - pad_top) / scale, 0, frame_height - 1)
            x2 = np.clip((boxes[:, 2] - pad_left) / scale, 0, frame_width - 1)
            y2 = np.clip((boxes[:, 3] - pad_top) / scale, 0, frame_height - 1)
            detections = [
                {
                    "class_name": self.names[int(class_ids[i])],
                    "confidence": round(float(scores[i]), 4),
                    "bbox": [round(float(x1[i]), 1), round(float(y1[i]), 1), round(float(x2[i]), 1), round(float(y2[i]), 1)],
                }
                for i in range(len(scores))
            ]
            return sorted(detections, key=lambda detection: detection["confidence"], reverse=True)

        class_count = len(self.names)
        if predictions.shape[0] in (4 + class_count, 5 + class_count):
            predictions = predictions.T
        feature_count = predictions.shape[1]
        if feature_count not in (4 + class_count, 5 + class_count):
            raise RuntimeError(
                f"Output ONNX có {feature_count} giá trị mỗi box, không khớp {class_count} class. "
                "Hãy export model detection với nms=False."
            )

        boxes = predictions[:, :4]
        if feature_count == 4 + class_count:
            class_scores = predictions[:, 4:]
        else:
            class_scores = predictions[:, 5:] * predictions[:, 4:5]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_scores)), class_ids]
        valid = scores >= confidence_threshold
        if not np.any(valid):
            return []

        boxes = boxes[valid]
        class_ids = class_ids[valid]
        scores = scores[valid]
        centers_x, centers_y, widths, heights = boxes.T
        x1 = (centers_x - widths / 2 - pad_left) / scale
        y1 = (centers_y - heights / 2 - pad_top) / scale
        x2 = (centers_x + widths / 2 - pad_left) / scale
        y2 = (centers_y + heights / 2 - pad_top) / scale
        x1 = np.clip(x1, 0, frame_width - 1)
        y1 = np.clip(y1, 0, frame_height - 1)
        x2 = np.clip(x2, 0, frame_width - 1)
        y2 = np.clip(y2, 0, frame_height - 1)

        detections = []
        for class_id in np.unique(class_ids):
            indexes = np.flatnonzero(class_ids == class_id)
            rects = [
                [int(x1[i]), int(y1[i]), max(1, int(x2[i] - x1[i])), max(1, int(y2[i] - y1[i]))]
                for i in indexes
            ]
            kept = cv2.dnn.NMSBoxes(rects, scores[indexes].astype(float).tolist(), confidence_threshold, 0.45)
            if len(kept) == 0:
                continue
            for local_index in np.asarray(kept).reshape(-1):
                i = indexes[int(local_index)]
                detections.append({
                    "class_name": self.names.get(int(class_ids[i]), str(int(class_ids[i]))),
                    "confidence": round(float(scores[i]), 4),
                    "bbox": [round(float(x1[i]), 1), round(float(y1[i]), 1), round(float(x2[i]), 1), round(float(y2[i]), 1)],
                })
        return sorted(detections, key=lambda detection: detection["confidence"], reverse=True)
