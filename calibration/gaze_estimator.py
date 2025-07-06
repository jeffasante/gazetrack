import numpy as np
import math
import joblib
from typing import Dict, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

class GazeEstimator:
    def __init__(self):
        self.calibration_data = []
        self.is_calibrated = False
        self.model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), Ridge(alpha=0.5))
        self.screen_width = 1920
        self.screen_height = 1080
        self.gaze_history = []
        self.max_history = 10
        
    def set_screen_dimensions(self, width: int, height: int):
        self.screen_width = width
        self.screen_height = height

    def _get_feature_vector(self, left_features: Dict, right_features: Dict) -> Optional[np.ndarray]:
        feature_keys = [
            'pupil_horizontal_ratio', 'pupil_vertical_ratio',
            'normalized_displacement_x', 'normalized_displacement_y', 'pupil_radius'
        ]
        vec = [left_features.get(k, 0.0) for k in feature_keys]
        vec.extend(right_features.get(k, 0.0) for k in feature_keys)
        return np.array(vec) if len(vec) == len(feature_keys) * 2 else None

    def add_calibration_point(self, screen_x: float, screen_y: float,
                            left_features: Dict, right_features: Dict):
        feature_vector = self._get_feature_vector(left_features, right_features)
        if feature_vector is not None:
            self.calibration_data.append({'screen_pos': (screen_x, screen_y), 'features': feature_vector})

    def train_regression_model(self) -> bool:
        if len(self.calibration_data) < 6:
            print("Need at least 6 calibration points for the model.")
            return False
        X = np.array([p['features'] for p in self.calibration_data])
        y = np.array([p['screen_pos'] for p in self.calibration_data])
        try:
            self.model.fit(X, y)
            self.is_calibrated = True
            print("Calibration model trained successfully!")
            return True
        except Exception as e:
            print(f"Failed to train regression model: {e}")
            return False

    def estimate_gaze_regression(self, left_features: Dict, right_features: Dict) -> Tuple[float, float]:
        if not self.is_calibrated: return self.estimate_gaze_geometric(left_features, right_features)
        feature_vector = self._get_feature_vector(left_features, right_features)
        if feature_vector is None: return self.estimate_gaze_geometric(left_features, right_features)
        gaze_point = self.model.predict(feature_vector.reshape(1, -1))[0]
        return (gaze_point[0] * 2) - 1, (gaze_point[1] * 2) - 1

    def estimate_gaze_geometric(self, left_features: Dict, right_features: Dict) -> Tuple[float, float]:
        avg_h = (left_features.get('pupil_horizontal_ratio', 0.5) + right_features.get('pupil_horizontal_ratio', 0.5)) / 2
        gaze_x = (avg_h - 0.5) * 3.0
        avg_v = (left_features.get('pupil_vertical_ratio', 0.5) + right_features.get('pupil_vertical_ratio', 0.5)) / 2
        gaze_y = (avg_v - 0.5) * 3.0
        return np.clip(gaze_x, -1, 1), np.clip(gaze_y, -1, 1)

    def estimate_gaze(self, left_landmarks, right_landmarks, left_pupil, right_pupil) -> Tuple[float, float]:
        left_features = self.extract_gaze_features(left_landmarks, left_pupil)
        right_features = self.extract_gaze_features(right_landmarks, right_pupil)
        gaze_x, gaze_y = self.estimate_gaze_regression(left_features, right_features) if self.is_calibrated else self.estimate_gaze_geometric(left_features, right_features)
        return self._apply_temporal_smoothing(gaze_x, gaze_y)

    def _apply_temporal_smoothing(self, gaze_x: float, gaze_y: float) -> Tuple[float, float]:
        self.gaze_history.append((gaze_x, gaze_y))
        if len(self.gaze_history) > self.max_history:
            self.gaze_history.pop(0)
        
        weights = np.linspace(0.1, 1.0, len(self.gaze_history))
        weights /= np.sum(weights)
        
        smoothed_x = sum(w * gx for w, (gx, gy) in zip(weights, self.gaze_history))
        smoothed_y = sum(w * gy for w, (gx, gy) in zip(weights, self.gaze_history))
        
        return smoothed_x, smoothed_y

    def gaze_to_screen_coordinates(self, gaze_x: float, gaze_y: float) -> Tuple[int, int]:
        norm_x, norm_y = (gaze_x + 1) / 2, (gaze_y + 1) / 2
        screen_x, screen_y = int(norm_x * self.screen_width), int(norm_y * self.screen_height)
        return np.clip(screen_x, 0, self.screen_width-1), np.clip(screen_y, 0, self.screen_height-1)
    
    def extract_gaze_features(self, landmarks: Dict, pupil_result: Optional[Dict]) -> Dict:
        features = {}
        if not landmarks: return features
        p, i, o = landmarks.get('pupil_center'), landmarks.get('inner_corner'), landmarks.get('outer_corner')
        ul, ll = landmarks.get('upper_lid_2'), landmarks.get('lower_lid_2')

        if p and i and o and ul and ll:
            eye_width = math.dist(i, o)
            eye_height = math.dist(ul, ll)
            if eye_width > 0: features['pupil_horizontal_ratio'] = 1.0 - ((p[0] - i[0]) / (o[0] - i[0]))
            if eye_height > 0: features['pupil_vertical_ratio'] = (p[1] - ul[1]) / (ll[1] - ul[1])
            eye_center_x, eye_center_y = (i[0]+o[0])/2, (ul[1]+ll[1])/2
            if eye_width > 0: features['normalized_displacement_x'] = -1 * (p[0] - eye_center_x) / (eye_width / 2)
            if eye_height > 0: features['normalized_displacement_y'] = (p[1] - eye_center_y) / (eye_height / 2)

        if pupil_result and 'area' in pupil_result:
            features['pupil_radius'] = math.sqrt(pupil_result['area'] / math.pi)
        return features

    def save_calibration(self, filename: str) -> bool:
        if not self.is_calibrated: 
            print("Model not calibrated. Nothing to save.")
            return False
        try:
            joblib.dump(self.model, filename)
            print(f"Calibration model saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_calibration(self, filename: str) -> bool:
        try:
            self.model = joblib.load(filename)
            self.is_calibrated = True
            print(f"Calibration model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_calibrated = False
            return False