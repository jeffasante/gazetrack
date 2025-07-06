import cv2
import numpy as np
from typing import List, Tuple

class FaceDetector:
    def __init__(self):
        cascade_path = 'models/haarcascade_frontalface_default.xml'
        try:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise IOError(f"Failed to load Haar Cascade model from {cascade_path}")
        except Exception as e:
            print(f"Error initializing FaceDetector: {e}")
            print("Please ensure 'haarcascade_frontalface_default.xml' is in 'models' directory.")
            raise
        self.previous_faces = []
        self.smoothing_alpha = 0.4

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        detected_faces = [tuple(face) for face in faces]
        smoothed_faces = self._smooth_detections(detected_faces)
        self.previous_faces = smoothed_faces
        return smoothed_faces

    def _smooth_detections(self, current_faces: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        if not current_faces:
            return []
        user_face = max(current_faces, key=lambda f: f[2] * f[3])
        if not self.previous_faces:
            return [user_face]
        prev_face = self.previous_faces[0]
        cx, cy, cw, ch = user_face
        px, py, pw, ph = prev_face
        smooth_x = int(self.smoothing_alpha * cx + (1 - self.smoothing_alpha) * px)
        smooth_y = int(self.smoothing_alpha * cy + (1 - self.smoothing_alpha) * py)
        smooth_w = int(self.smoothing_alpha * cw + (1 - self.smoothing_alpha) * pw)
        smooth_h = int(self.smoothing_alpha * ch + (1 - self.smoothing_alpha) * ph)
        return [(smooth_x, smooth_y, smooth_w, smooth_h)]