import dlib
import cv2
import numpy as np
from typing import Dict, List, Tuple

class DlibDetector:
    def __init__(self):
        model_path = 'models/shape_predictor_68_face_landmarks.dat'
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(model_path)
        except Exception as e:
            print(f"Error loading dlib models: {e}")
            print(f"Please ensure 'shape_predictor_68_face_landmarks.dat' is in the 'models' directory.")
            raise

        # These are the indices for the 68-point model that correspond to the eyes
        self.EYE_LANDMARK_INDICES = {
            "left_eye": list(range(36, 42)),
            "right_eye": list(range(42, 48))
        }

    def _shape_to_np(self, shape, dtype="int") -> np.ndarray:
        """Converts dlib's shape object to a NumPy array."""
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detects faces and their landmarks in a frame.

        Returns:
            A list of dictionaries, one for each detected face. Each dictionary
            contains the face bounding box ('face_rect') and a dictionary of
            all 68 landmarks ('landmarks').
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        
        results = []
        for face in faces:
            shape = self.predictor(gray, face)
            landmarks = self._shape_to_np(shape)
            
            # Get the bounding box for the eye regions from the landmarks
            left_eye_pts = landmarks[self.EYE_LANDMARK_INDICES["left_eye"]]
            right_eye_pts = landmarks[self.EYE_LANDMARK_INDICES["right_eye"]]

            results.append({
                'face_rect': face,
                'landmarks': landmarks,
                'left_eye_pts': left_eye_pts,
                'right_eye_pts': right_eye_pts
            })
        return results