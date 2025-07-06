# tracking/pupil_tracker.py

import cv2
import numpy as np
import math
from typing import Dict, Optional

class PupilTracker:
    def __init__(self):
        self.previous_pupils = {}
        self.smoothing_alpha = 0.4

    def track_pupil(self, eye_region: np.ndarray, eye_name: str = '') -> Optional[Dict]:
        if eye_region.size == 0:
            return None

        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
        blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
        
        # This threshold is calculated based on the darkest part of the blurred eye.
        threshold_value = int(np.min(blurred_eye)) + 25
        _, binary_eye = cv2.threshold(blurred_eye, threshold_value, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Clean up the binary image to remove noise and close gaps
        cleaned_binary = cv2.morphologyEx(binary_eye, cv2.MORPH_OPEN, kernel)
        cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(best_contour)
        if M['m00'] == 0:
            return None
            
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        area = M['m00']
        
        perimeter = cv2.arcLength(best_contour, True)
        if perimeter == 0: return None
        circularity = 4 * math.pi * area / (perimeter**2)
        
        # Filter out shapes that are not circular enough to be a pupil
        if circularity < 0.6:
            return None

        pupil_data = {'center': (cx, cy), 'area': area, 'contour': best_contour}
        
        if eye_name:
            pupil_data = self._apply_temporal_smoothing(pupil_data, eye_name)
            
        return pupil_data

    def _apply_temporal_smoothing(self, current_pupil: Dict, eye_name: str) -> Dict:
        if eye_name not in self.previous_pupils or self.previous_pupils[eye_name] is None:
            self.previous_pupils[eye_name] = current_pupil
            return current_pupil
            
        prev_pupil = self.previous_pupils[eye_name]
        smooth_cx = int(self.smoothing_alpha * current_pupil['center'][0] + (1 - self.smoothing_alpha) * prev_pupil['center'][0])
        smooth_cy = int(self.smoothing_alpha * current_pupil['center'][1] + (1 - self.smoothing_alpha) * prev_pupil['center'][1])
        
        smoothed_pupil = current_pupil.copy()
        smoothed_pupil['center'] = (smooth_cx, smooth_cy)
        
        self.previous_pupils[eye_name] = smoothed_pupil
        return smoothed_pupil