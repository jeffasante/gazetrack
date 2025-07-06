import math
import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple

class CalibrationInterface:
    def __init__(self, gaze_estimator, screen_width: int, screen_height: int):
        self.gaze_estimator = gaze_estimator
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.calibration_points = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
        ]
        self.current_point_index = 0
        self.is_calibrating = False
        self.samples_per_point = 30
        self.current_samples = []
        self.collection_start_time = 0
        self.point_display_time = 1.0

    def start_calibration(self):
        self.is_calibrating = True
        self.current_point_index = 0
        self.gaze_estimator.calibration_data = []
        self.current_samples = []
        self.collection_start_time = time.time()
        print(f"Starting calibration with {len(self.calibration_points)} points...")

    def is_calibration_active(self) -> bool:
        return self.is_calibrating

    def process_calibration_frame(self, frame: np.ndarray,
                                left_landmarks: Dict, right_landmarks: Dict,
                                left_pupil: Optional[Dict], right_pupil: Optional[Dict]) -> np.ndarray:
        """
        Process frame during calibration.
        """
        if not self.is_calibrating:
            return frame

        # Are we getting all the data we need?
        # The main loop ensures these are not passed if no face is detected.
        is_ready = bool(left_landmarks and right_landmarks and left_pupil and right_pupil)

        # Create the UI overlay
        overlay = self._create_calibration_overlay(frame.shape, is_ready)
        result = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

        # Collect data only if detection is good and display time has passed
        if is_ready and time.time() - self.collection_start_time > self.point_display_time:
            left_features = self.gaze_estimator.extract_gaze_features(left_landmarks, left_pupil)
            right_features = self.gaze_estimator.extract_gaze_features(right_landmarks, right_pupil)

            if left_features and right_features:
                self.current_samples.append({'left': left_features, 'right': right_features})

        # Check if we have collected enough samples for the current point
        if len(self.current_samples) >= self.samples_per_point:
            self._finalize_current_point()

        # Draw UI text on the screen
        progress = f"Point {self.current_point_index + 1}/{len(self.calibration_points)}"
        samples_info = f"Samples: {len(self.current_samples)}/{self.samples_per_point}"
        status_msg = "Look at the point" if is_ready else "Position face in center"
        
        cv2.putText(result, progress, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(result, samples_info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(result, status_msg, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_ready else (0, 0, 255), 2)

        return result

    def _create_calibration_overlay(self, frame_shape: Tuple, is_ready: bool) -> np.ndarray:
        h, w, _ = frame_shape
        overlay = np.full((h, w, 3), (30,30,30), dtype=np.uint8)
        if self.current_point_index >= len(self.calibration_points): return overlay
        
        norm_x, norm_y = self.calibration_points[self.current_point_index]
        point = (int(norm_x * w), int(norm_y * h))
        
        # Point is green if ready, red if not
        color = (0, 255, 0) if is_ready else (0, 0, 255)
        
        # Pulsing effect during collection
        if is_ready and time.time() - self.collection_start_time > self.point_display_time:
             pulse = 0.6 + 0.4 * abs(math.sin(time.time() * 5))
             color = (0, int(255 * pulse), 0)

        cv2.circle(overlay, point, 20, color, -1)
        cv2.circle(overlay, point, 25, (255, 255, 255), 2)
        return overlay

    def _finalize_current_point(self):
        if not self.current_samples: 
            print(f"Point {self.current_point_index + 1}: No samples collected, skipping.")
        else:
            norm_x, norm_y = self.calibration_points[self.current_point_index]
            avg_left = self._average_features([s['left'] for s in self.current_samples])
            avg_right = self._average_features([s['right'] for s in self.current_samples])
            self.gaze_estimator.add_calibration_point(norm_x, norm_y, avg_left, avg_right)
            print(f"Completed point {self.current_point_index + 1} with {len(self.current_samples)} samples.")
        
        self.current_point_index += 1
        self.current_samples = []
        self.collection_start_time = time.time()
        
        if self.current_point_index >= len(self.calibration_points):
            self._complete_calibration()

    def _average_features(self, feature_list: List[Dict]) -> Dict:
        if not feature_list: return {}
        # Ensure we handle cases where a feature might be missing in a frame
        avg = {}
        all_keys = set(k for d in feature_list for k in d)
        for key in all_keys:
            values = [d.get(key, 0.0) for d in feature_list]
            avg[key] = sum(values) / len(values)
        return avg

    def _complete_calibration(self):
        print("Calibration data collection complete!")
        self.is_calibrating = False
        self.gaze_estimator.train_regression_model()