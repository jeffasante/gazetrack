import cv2
import numpy as np
import time
import math
from typing import Tuple

class Validator:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.validation_points = [
            (0.25, 0.25), (0.75, 0.25),
            (0.25, 0.75), (0.75, 0.75),
            (0.5, 0.5),
        ]
        self.is_validating = False
        self.current_point_index = 0
        self.samples_per_point = 50
        self.current_gaze_samples = []
        self.validation_results = []
        self.collection_start_time = 0
        self.point_display_time = 1.0

    def start(self):
        if self.is_validating: return
        print("\n= Starting Validation =")
        self.is_validating = True
        self.current_point_index = 0
        self.validation_results = []
        self.current_gaze_samples = []
        self.collection_start_time = time.time()

    def stop(self):
        if not self.is_validating: return
        self.is_validating = False
        print("= Validation Complete =")
        self._calculate_and_print_metrics()

    def _calculate_and_print_metrics(self):
        if not self.validation_results:
            print("No validation data collected.")
            return
        errors_pixels = [res['error_pixels'] for res in self.validation_results]
        mean_error = np.mean(errors_pixels)
        std_dev_error = np.std(errors_pixels)
        max_error = np.max(errors_pixels)
        print(f"Validation Points: {len(errors_pixels)}")
        print(f"Mean Error: {mean_error:.2f} px")
        print(f"Std Dev: {std_dev_error:.2f} px")
        print(f"Max Error: {max_error:.2f} px\n")

    def _finalize_current_point(self):
        if self.current_gaze_samples:
            target_norm_x, target_norm_y = self.validation_points[self.current_point_index]
            target_px = int(target_norm_x * self.screen_width)
            target_py = int(target_norm_y * self.screen_height)
            avg_gaze_x = int(np.mean([s[0] for s in self.current_gaze_samples]))
            avg_gaze_y = int(np.mean([s[1] for s in self.current_gaze_samples]))
            error = math.dist((target_px, target_py), (avg_gaze_x, avg_gaze_y))
            self.validation_results.append({'error_pixels': error})
            print(f"Point {self.current_point_index + 1}: Error = {error:.2f} pixels")
        
        self.current_point_index += 1
        self.current_gaze_samples = []
        self.collection_start_time = time.time()
        if self.current_point_index >= len(self.validation_points):
            self.stop()

    def process_frame(self, frame: np.ndarray, estimated_gaze_px: Tuple[int, int]) -> np.ndarray:
        if not self.is_validating: return frame
        overlay = self._create_validation_overlay(frame.shape)
        result = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        
        if time.time() - self.collection_start_time > self.point_display_time and estimated_gaze_px:
            self.current_gaze_samples.append(estimated_gaze_px)
        
        if len(self.current_gaze_samples) >= self.samples_per_point:
            self._finalize_current_point()
        
        if self.is_validating:
            progress = f"Validating Point {self.current_point_index + 1}/{len(self.validation_points)}"
            samples_info = f"Samples: {len(self.current_gaze_samples)}/{self.samples_per_point}"
            cv2.putText(result, progress, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(result, samples_info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        return result

    def _create_validation_overlay(self, frame_shape: Tuple) -> np.ndarray:
        h, w, _ = frame_shape
        overlay = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)
        if self.current_point_index >= len(self.validation_points): return overlay
        norm_x, norm_y = self.validation_points[self.current_point_index]
        point = (int(norm_x * w), int(norm_y * h))
        cv2.circle(overlay, point, 20, (255, 0, 255), -1)
        cv2.circle(overlay, point, 25, (255, 255, 255), 2)
        return overlay