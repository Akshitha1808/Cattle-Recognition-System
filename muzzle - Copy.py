import os 
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# -------------------- Load Dataset --------------------
def load_muzzle_dataset(base_dir, img_size=(100, 100)):
    images = []
    labels = []

    print(f"\n📁 Loading data from: {base_dir}")
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"❌ Dataset directory not found: {base_dir}")

    for name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, name)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            filepath = os.path.join(class_path, filename)
            img = cv2.imread(filepath)
            if img is None:
                print(f"⚠️ Skipping unreadable image: {filepath}")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, img_size)
            images.append(resized.flatten())
            labels.append(name)

    print(f"✅ Loaded {len(images)} images across {len(set(labels))} labels.")
    return np.array(images), np.array(labels)

# -------------------- Train Model --------------------
def train_muzzle_recognition_model(X, y):
    if len(X) == 0 or len(y) == 0:
        raise ValueError("❌ No data provided for training. Check dataset path and image loading.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"✅ Model trained. Accuracy: {acc * 100:.2f}%")

    joblib.dump(model, "muzzle_model.pkl")
    joblib.dump(label_encoder, "muzzle_labels.pkl")
    print("💾 Model and labels saved.")

# -------------------- Predict --------------------
def predict_muzzle(image_path, img_size=(100, 100)):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    model = joblib.load("muzzle_model.pkl")
    label_encoder = joblib.load("muzzle_labels.pkl")

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, img_size)
    features = resized.flatten().reshape(1, -1)

    pred = model.predict(features)
    prob = model.predict_proba(features)
    name = label_encoder.inverse_transform(pred)[0]
    confidence = np.max(prob)

    print(f"🔍 Predicted: {name} ({confidence * 100:.2f}%)")

    cv2.putText(img, f"{name} ({confidence * 100:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------- Main Pipeline --------------------
if __name__ == "__main__":
    dataset_dir = r"C:\Users\akshi\OneDrive\Desktop\muzzle-recognition\BeefCattle_Muzzle_database"
    print("📂 Looking for data at:", dataset_dir)

    X, y = load_muzzle_dataset(dataset_dir)
    train_muzzle_recognition_model(X, y)

    # 🧪 Example prediction — change path to test image
    predict_muzzle(r"C:\Users\akshi\OneDrive\Desktop\muzzle-recognition\BeefCattle_Muzzle_database\cattle_4545\cattle_4545_DSCF0955.jpg")
