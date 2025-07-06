from setuptools import setup, find_packages

setup(
    name="gazetrack",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy", 
        "dlib",
        "scikit-learn",
        "joblib"
    ]
)