import cv2
import numpy as np
from typing import Dict, Tuple

class EyeDetector:
    def __init__(self):
        eye_cascade_path = 'models/haarcascade_eye.xml'
        try:
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            if self.eye_cascade.empty():
                raise IOError(f"Failed to load Haar Cascade model from {eye_cascade_path}")
        except Exception as e:
            print(f"Error initializing EyeDetector: {e}")
            print("Please ensure 'haarcascade_eye.xml' is in the 'models' directory.")
            raise

    def extract_eyes_from_face(self, frame: np.ndarray, 
                             face_coords: Tuple[int, int, int, int]) -> Dict:
        fx, fy, fw, fh = face_coords
        face_roi_gray = cv2.cvtColor(frame[fy:fy+fh, fx:fx+fw], cv2.COLOR_BGR2GRAY)
        
        eyes = self.eye_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(25, 25)
        )

        if len(eyes) != 2:
            return {'left_eye': {'is_valid': False}, 'right_eye': {'is_valid': False}}

        eyes = sorted(eyes, key=lambda e: e[0])
        right_eye_local, left_eye_local = eyes[0], eyes[1]
        
        eye_data = {}
        for eye_name, local_coords in [('left_eye', left_eye_local), ('right_eye', right_eye_local)]:
            ex_local, ey_local, ew, eh = local_coords
            ex_abs, ey_abs = fx + ex_local, fy + ey_local
            eye_region_color = frame[ey_abs:ey_abs+eh, ex_abs:ex_abs+ew]
            eye_data[eye_name] = {
                'region': eye_region_color,
                'coords': (ex_abs, ey_abs, ew, eh),
                'is_valid': True
            }
        return eye_data