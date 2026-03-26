import cv2
import joblib
import os

from src.feature_extraction import extract_features

def predict(path):
    # Load model + scaler
    model = joblib.load("models/svm_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    labels = ["anthracnose", "powdery_mildew", "healthy"]

    # Load image
    path = os.path.join("for_predict", path)
    img = cv2.imread(path)
    img = cv2.resize(img, (120, 150), interpolation=cv2.INTER_AREA)

    # Extract features
    features = extract_features(img)

    # Scale
    features = scaler.transform([features])

    # Predict
    pred = model.predict(features)[0]

    print("Prediction:", labels[pred])