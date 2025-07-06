import cv2
import os
import urllib.request
import bz2

def setup_models():
    os.makedirs('models', exist_ok=True)
    
    # Copy Haar cascades from OpenCV
    cv2_data = cv2.data.haarcascades
    eye_cascade = os.path.join(cv2_data, 'haarcascade_eye.xml')
    
    import shutil
    shutil.copy(eye_cascade, 'models/')
    
    # Download dlib model if not exists
    if not os.path.exists('models/shape_predictor_68_face_landmarks.dat'):
        print("Downloading dlib facial landmarks model...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        urllib.request.urlretrieve(url, 'models/shape_predictor_68_face_landmarks.dat.bz2')
        
        with bz2.BZ2File('models/shape_predictor_68_face_landmarks.dat.bz2') as f:
            with open('models/shape_predictor_68_face_landmarks.dat', 'wb') as out:
                out.write(f.read())
        
        os.remove('models/shape_predictor_68_face_landmarks.dat.bz2')
        print("Model setup complete!")

if __name__ == "__main__":
    setup_models()