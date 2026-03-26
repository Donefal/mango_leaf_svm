import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from src.preprocess import load_and_resize
from src.feature_extraction import extract_features

def make_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Anthracnose", "Powdery Mildew", "Healthy"]

    plt.imshow(cm)
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.show()

def load_dataset(dataset_path):
    X = []
    y = []

    labels = {
        "anthracnose": 0,
        "powdery_mildew": 1,
        "healthy": 2
    }

    for label_name in labels:
        folder = os.path.join(dataset_path, label_name)

        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            img = load_and_resize(path)
            features = extract_features(img)

            X.append(features)
            y.append(labels[label_name])
    return np.array(X), np.array(y)


def train(file_path):
    print("Loading dataset...")
    X, y = load_dataset(file_path)

    print("Dataset shape:", X.shape)  # (num_samples, 1740)

    # ---------------------------
    # TRAIN-TEST SPLIT
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------
    # SCALING (VERY IMPORTANT)
    # ---------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------------------
    # TRAIN SVM
    # ---------------------------
    print("Training SVM...")
    model = svm.SVC(kernel='rbf')  # good default

    model.fit(X_train, y_train)

    # ---------------------------
    # EVALUATION
    # ---------------------------
    print("Evaluating...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ---------------------------
    # SAVE MODEL + SCALER
    # ---------------------------
    joblib.dump(model, "models/svm_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    make_confusion_matrix(y_pred, y_test)
