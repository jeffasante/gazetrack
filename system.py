# main.py

import cv2
import numpy as np
from capture import CameraCapture
from detection.dlib_detector import DlibDetector
from pupil_tracker import PupilTracker
from calibration.gaze_estimator import GazeEstimator
from calibration.calibration_interface import CalibrationInterface
from validator import Validator

class GazePySystem:
    def __init__(self, screen_width=1920, screen_height=1080):
        self.cam = CameraCapture()
        self.dlib_detector = DlibDetector()
        self.pupil_tracker = PupilTracker()
        self.gaze_estimator = GazeEstimator()
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.gaze_estimator.set_screen_dimensions(self.screen_width, self.screen_height)
        
        self.calibration_interface = CalibrationInterface(self.gaze_estimator, self.screen_width, self.screen_height)
        self.validator = Validator(self.screen_width, self.screen_height)
        self.show_debug_overlays = False

    def _get_eye_region(self, frame, eye_points):
        x_min, y_min = np.min(eye_points, axis=0)
        x_max, y_max = np.max(eye_points, axis=0)
        padding_x = int((x_max - x_min) * 0.25)
        padding_y = int((y_max - y_min) * 0.5)
        x_min, y_min = max(0, x_min - padding_x), max(0, y_min - padding_y)
        x_max, y_max = min(frame.shape[1], x_max + padding_x), min(frame.shape[0], y_max + padding_y)
        eye_region_frame = frame[y_min:y_max, x_min:x_max]
        coords = (x_min, y_min, x_max-x_min, y_max-y_min)
        return eye_region_frame, coords

    def _extract_landmarks_for_gaze(self, eye_points, eye_bbox_coords):
        bbox_x, bbox_y = eye_bbox_coords[0], eye_bbox_coords[1]
        inner_corner = tuple(eye_points[0])
        outer_corner = tuple(eye_points[3])
        upper_lid = tuple(min(eye_points, key=lambda p: p[1]))
        lower_lid = tuple(max(eye_points, key=lambda p: p[1]))
        return {
            "inner_corner": (inner_corner[0] - bbox_x, inner_corner[1] - bbox_y),
            "outer_corner": (outer_corner[0] - bbox_x, outer_corner[1] - bbox_y),
            "upper_lid_2":  (upper_lid[0]  - bbox_x, upper_lid[1]  - bbox_y),
            "lower_lid_2":  (lower_lid[0]  - bbox_x, lower_lid[1]  - bbox_y),
        }

    def process_frame(self, frame):
        data = {}
        detections = self.dlib_detector.detect(frame)
        if detections:
            d = detections[0]
            data['dlib_detection'] = d
            left_eye_region, left_eye_coords = self._get_eye_region(frame, d['left_eye_pts'])
            right_eye_region, right_eye_coords = self._get_eye_region(frame, d['right_eye_pts'])
            data['left_eye_coords'], data['right_eye_coords'] = left_eye_coords, right_eye_coords
            left_pupil = self.pupil_tracker.track_pupil(left_eye_region, 'left_eye')
            right_pupil = self.pupil_tracker.track_pupil(right_eye_region, 'right_eye')
            data['left_pupil'], data['right_pupil'] = left_pupil, right_pupil
            left_landmarks = self._extract_landmarks_for_gaze(d['left_eye_pts'], left_eye_coords)
            if left_pupil: left_landmarks['pupil_center'] = left_pupil['center']
            right_landmarks = self._extract_landmarks_for_gaze(d['right_eye_pts'], right_eye_coords)
            if right_pupil: right_landmarks['pupil_center'] = right_pupil['center']
            data['left_landmarks'], data['right_landmarks'] = left_landmarks, right_landmarks
            if left_landmarks.get('pupil_center') and right_landmarks.get('pupil_center'):
                data['gaze_coords'] = self.gaze_estimator.estimate_gaze(left_landmarks, right_landmarks, left_pupil, right_pupil)
        return frame, data

    def _draw_debug_visualization(self, frame, data):
        if 'dlib_detection' not in data: return
        d = data['dlib_detection']
        for (x, y) in d['landmarks']:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        if 'left_eye_coords' in data:
            x,y,w,h = data['left_eye_coords']
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 1)
        if 'right_eye_coords' in data:
            x,y,w,h = data['right_eye_coords']
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 1)

    def run_interactive(self):
        if not self.cam.start(): return
        print("\n--- GazePy System (dlib backend) ---")
        print("  'd' - Toggle dlib Landmark Visualization")
        print("  'c' - Start Calibration")
        print("  'v' - Start Validation (after calibration)")
        print("  'l' - Load Calibration ('calibration_model.joblib')")
        print("  's' - Save Calibration ('calibration_model.joblib')")
        print("  'q' or ESC - Quit")
        print("--------------------------------------")
        try:
            while True:
                frame = self.cam.get_frame()
                if frame is None: break
                _, data = self.process_frame(frame)
                result = frame.copy()

                if self.calibration_interface.is_calibration_active():
                    result = self.calibration_interface.process_calibration_frame(
                        frame,
                        data.get('left_landmarks',{}),
                        data.get('right_landmarks',{}),
                        data.get('left_pupil'),
                        data.get('right_pupil')
                    )
                elif self.validator.is_validating:
                    gaze_px = self.gaze_estimator.gaze_to_screen_coordinates(*data['gaze_coords']) if 'gaze_coords' in data else None
                    result = self.validator.process_frame(frame, gaze_px)
                else:
                    if 'gaze_coords' in data:
                        screen_gaze_px = self.gaze_estimator.gaze_to_screen_coordinates(*data['gaze_coords'])
                        h, w, _ = result.shape
                        frame_gaze_x = int(screen_gaze_px[0] * w / self.screen_width)
                        frame_gaze_y = int(screen_gaze_px[1] * h / self.screen_height)
                        cv2.circle(result, (frame_gaze_x, frame_gaze_y), 15, (0, 255, 255), 3)
                        cv2.circle(result, (frame_gaze_x, frame_gaze_y), 8, (0, 0, 255), -1)
                
                if self.show_debug_overlays:
                    self._draw_debug_visualization(result, data)
                
                cv2.imshow('GazePy System', result)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: break
                elif key == ord('d'): self.show_debug_overlays = not self.show_debug_overlays
                elif key == ord('c'): self.calibration_interface.start_calibration()
                elif key == ord('v'): 
                    if self.gaze_estimator.is_calibrated: self.validator.start()
                    else: print("\n[INFO] Please calibrate the system first by pressing 'c'.")
                elif key == ord('l'): self.gaze_estimator.load_calibration('calibration_model.joblib')
                elif key == ord('s'): self.gaze_estimator.save_calibration('calibration_model.joblib')
        finally:
            self.cam.release()

if __name__ == "__main__":
    system = GazePySystem()
    system.run_interactive()