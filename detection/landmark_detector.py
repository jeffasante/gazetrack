import cv2
import numpy as np
from typing import Tuple, Optional, Dict

class LandmarkDetector:
    def __init__(self):
        self.previous_landmarks = {'left_eye': None, 'right_eye': None}
        self.smoothing_alpha = 0.4

    def _get_eye_boundary_landmarks(self, eye_region: np.ndarray) -> Dict:
        if eye_region.size == 0: return {}
        
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 80)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return {}

        all_points = np.concatenate(contours).reshape(-1, 2)

        landmarks = {}
        landmarks['inner_corner'] = tuple(all_points[np.argmin(all_points[:, 0])])
        landmarks['outer_corner'] = tuple(all_points[np.argmax(all_points[:, 0])])
        landmarks['upper_lid_2'] = tuple(all_points[np.argmin(all_points[:, 1])])
        landmarks['lower_lid_2'] = tuple(all_points[np.argmax(all_points[:, 1])])
        
        return landmarks

    def _smooth_landmarks(self, current_landmarks: Dict, eye_name: str) -> Dict:
        if eye_name not in self.previous_landmarks or self.previous_landmarks[eye_name] is None:
            self.previous_landmarks[eye_name] = current_landmarks
            return current_landmarks

        smoothed = {}
        prev_landmarks = self.previous_landmarks[eye_name]
        for key, current_pos in current_landmarks.items():
            prev_pos = prev_landmarks.get(key, current_pos)
            smooth_x = int(self.smoothing_alpha * current_pos[0] + (1 - self.smoothing_alpha) * prev_pos[0])
            smooth_y = int(self.smoothing_alpha * current_pos[1] + (1 - self.smoothing_alpha) * prev_pos[1])
            smoothed[key] = (smooth_x, smooth_y)
        
        self.previous_landmarks[eye_name] = smoothed
        return smoothed

    def detect_all_landmarks(self, eye_region: np.ndarray, 
                           pupil_result: Optional[Dict] = None,
                           eye_name: str = 'left_eye') -> Dict[str, Tuple[int, int]]:
        landmarks = self._get_eye_boundary_landmarks(eye_region)
        
        if pupil_result and 'center' in pupil_result:
            landmarks['pupil_center'] = pupil_result['center']
        
        if not all(k in landmarks for k in ['inner_corner', 'outer_corner', 'upper_lid_2', 'lower_lid_2']):
             return self.previous_landmarks.get(eye_name, {})

        smoothed_landmarks = self._smooth_landmarks(landmarks, eye_name)
        return smoothed_landmarks