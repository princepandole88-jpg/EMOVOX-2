import threading
import time
from collections import Counter, deque

import cv2
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
except Exception:
    mp = None
    HAS_MEDIAPIPE = False


class FaceEmotionMonitor:
    """Continuous background webcam face & expression detector."""

    def __init__(
        self,
        camera_index=0,
        on_update=None,
        on_sad_detected=None,
        sad_trigger_sec=4.0,
        sad_cooldown_sec=180.0,
    ):
        self.camera_index = camera_index
        self.on_update = on_update
        self.on_sad_detected = on_sad_detected
        self.sad_trigger_sec = sad_trigger_sec
        self.sad_cooldown_sec = sad_cooldown_sec

        self.running = False
        self.thread = None
        self.cap = None

        self.face_detected = False
        self.expression = "neutral"
        self.is_sad = False
        self.confidence = 0.0

        self._history = deque(maxlen=12)
        self._sad_since = None
        self._last_sad_alert = 0.0

        self._face_mesh = None
        if HAS_MEDIAPIPE:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )
        self._eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _classify_landmarks(self, lm):
        def pt(idx):
            return np.array([lm[idx].x, lm[idx].y], dtype=np.float32)

        left_corner = pt(61)
        right_corner = pt(291)
        upper_lip = pt(13)
        lower_lip = pt(14)
        mouth_center = (upper_lip + lower_lip) / 2.0

        left_eye_outer = pt(33)
        right_eye_outer = pt(263)
        inter_ocular = float(np.linalg.norm(right_eye_outer - left_eye_outer)) + 1e-6

        mouth_width = float(np.linalg.norm(right_corner - left_corner)) / inter_ocular
        mouth_open = float(np.linalg.norm(lower_lip - upper_lip)) / inter_ocular
        corners_y = (left_corner[1] + right_corner[1]) / 2.0
        smile_lift = (mouth_center[1] - corners_y) / inter_ocular

        eye_y = (pt(159)[1] + pt(386)[1]) / 2.0
        inner_brow_y = (pt(107)[1] + pt(336)[1]) / 2.0
        outer_brow_y = (pt(70)[1] + pt(300)[1]) / 2.0
        inner_brow_dist = float(np.linalg.norm(pt(336) - pt(107))) / inter_ocular

        inner_brow_tilt = (outer_brow_y - inner_brow_y) / inter_ocular
        brow_height = (eye_y - inner_brow_y) / inter_ocular

        if smile_lift > 0.032 or (mouth_width > 0.62 and smile_lift > 0.015):
            return "happy", min(0.95, 0.60 + smile_lift * 6.0)

        if inner_brow_dist < 0.28 and brow_height < 0.145 and mouth_open < 0.04:
            return "frustrated", 0.78

        if (smile_lift < -0.018 and inner_brow_tilt > 0.012) or (smile_lift < -0.026):
            conf = min(0.95, 0.62 + abs(smile_lift) * 8.0 + max(0.0, inner_brow_tilt) * 4.0)
            return "sad", conf

        if inner_brow_dist < 0.30 and brow_height < 0.165:
            return "stressed", 0.72

        return "neutral", 0.68

    def _classify_opencv(self, gray):
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return False, "no_face", 0.0

        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        face_roi = gray[y : y + h, x : x + w]
        lower_roi = face_roi[int(h * 0.52) : int(h * 0.92), int(w * 0.18) : int(w * 0.82)]

        smiles = self._smile_cascade.detectMultiScale(
            lower_roi, scaleFactor=1.7, minNeighbors=22
        )
        if len(smiles) > 0:
            return True, "happy", 0.78

        if lower_roi.size > 0:
            blur = cv2.GaussianBlur(lower_roi, (5, 5), 0)
            sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.abs(sobel_y)
            lh, lw = grad.shape
            if lh > 12 and lw > 18:
                left_band = grad[:, : lw // 3]
                mid_band = grad[:, lw // 3 : 2 * lw // 3]
                right_band = grad[:, 2 * lw // 3 :]

                y_idx = np.arange(lh, dtype=np.float32)[:, None]
                y_left = float((left_band * y_idx).sum() / (left_band.sum() + 1e-5)) / lh
                y_mid = float((mid_band * y_idx).sum() / (mid_band.sum() + 1e-5)) / lh
                y_right = float((right_band * y_idx).sum() / (right_band.sum() + 1e-5)) / lh

                corners_y = (y_left + y_right) / 2.0
                droop = corners_y - y_mid

                eyes = self._eye_cascade.detectMultiScale(
                    face_roi[: int(h * 0.55), :], scaleFactor=1.15, minNeighbors=5
                )
                eye_y_ratio = (
                    float(np.mean([ey + eh * 0.5 for (_, ey, _, eh) in eyes])) / h
                    if len(eyes) > 0
                    else 0.34
                )

                if droop > 0.055 or (droop > 0.035 and eye_y_ratio > 0.39):
                    conf = min(0.88, 0.60 + droop * 2.5)
                    return True, "sad", conf

        return True, "neutral", 0.65

    def _loop(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(1.0)
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.1)
                continue

            detected = False
            raw_expr = "neutral"
            raw_conf = 0.0

            if self._face_mesh is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self._face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    detected = True
                    lm = res.multi_face_landmarks[0].landmark
                    raw_expr, raw_conf = self._classify_landmarks(lm)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected, raw_expr, raw_conf = self._classify_opencv(gray)

            self.face_detected = detected
            if detected:
                self._history.append(raw_expr)
                most_common, count = Counter(self._history).most_common(1)[0]
                self.expression = most_common
                self.confidence = round(max(raw_conf, count / len(self._history)), 2)
                self.is_sad = self.expression == "sad"
            else:
                self._history.clear()
                self.expression = "no_face"
                self.is_sad = False
                self.confidence = 0.0

            now = time.time()
            if self.is_sad:
                if self._sad_since is None:
                    self._sad_since = now
                elif (now - self._sad_since) >= self.sad_trigger_sec and (
                    now - self._last_sad_alert
                ) >= self.sad_cooldown_sec:
                    self._last_sad_alert = now
                    if callable(self.on_sad_detected):
                        try:
                            self.on_sad_detected()
                        except Exception:
                            pass
            else:
                self._sad_since = None

            if callable(self.on_update):
                try:
                    self.on_update(
                        self.face_detected, self.expression, self.is_sad, self.confidence
                    )
                except Exception:
                    pass

            time.sleep(0.12)
