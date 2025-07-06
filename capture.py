import cv2
import numpy as np
import time
from typing import Optional, Tuple, Callable

class CameraCapture:
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
    
    def start(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_id}")
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera started: {actual_width}x{actual_height}")
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            self._calculate_fps()
            return frame
        return None

    def _calculate_fps(self):
        if self.frame_count % 30 == 0:
            current_time = time.time()
            elapsed = current_time - self.start_time
            if elapsed > 0:
                self.fps = self.frame_count / elapsed
    
    def get_fps(self) -> float:
        return self.fps

    def release(self):
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Camera released")