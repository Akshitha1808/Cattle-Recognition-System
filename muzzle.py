import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import joblib
from skimage.feature import hog, local_binary_pattern
from skimage import exposure

# -------------------- FEATURE EXTRACTION --------------------
def extract_hog_features(image):
    image = exposure.equalize_hist(image)
    return hog(image, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), visualize=False)

def extract_lbp_features(image):
    radius = 3
    n_points = 8
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, n_points + 3),
                           range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_sift_features(image):
    sift = cv2.SIFT_create(nfeatures=100)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is not None:
        return np.mean(descriptors, axis=0)
    return np.zeros(128)

def enhance_image(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return cv2.bilateralFilter(blurred, 9, 75, 75)

# -------------------- LOAD DATASET --------------------
def load_dataset(base_dir, img_size=(64, 64)):
    features = []
    labels = []

    print(f"\n📁 Loading data from: {base_dir}")

    for name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, name)
        if not os.path.isdir(class_path):
            continue

        print(f"Processing {name}")

        for file in os.listdir(class_path):
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            path = os.path.join(class_path, file)
            img = cv2.imread(path)

            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, img_size)
            enhanced = enhance_image(resized)

            hog_feat = extract_hog_features(enhanced)
            lbp_feat = extract_lbp_features(enhanced)
            sift_feat = extract_sift_features(enhanced)

            combined = np.concatenate([hog_feat, lbp_feat, sift_feat])

            features.append(combined)
            labels.append(name)

    print(f"✅ Loaded {len(features)} samples, {len(set(labels))} classes")
    return np.array(features), np.array(labels)

# -------------------- TRAIN MODEL --------------------
def train_model(X, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Optimized PCA
    pca = PCA(n_components=80)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    print(f"📊 Features after PCA: {X_train.shape[1]}")

    # Stable models
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    knn = KNeighborsClassifier(n_neighbors=5)

    svm.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    print("\n📈 Model Accuracy:")
    print(f"SVM: {svm.score(X_test, y_test):.3f}")
    print(f"KNN: {knn.score(X_test, y_test):.3f}")

    # Use best model (KNN is stable)
    model = knn

    # Save
    joblib.dump(model, "model.pkl")
    joblib.dump(label_encoder, "labels.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(pca, "pca.pkl")

    print("💾 Model saved successfully")

# -------------------- PREDICTION --------------------
def predict(image_path, img_size=(64, 64)):
    model = joblib.load("model.pkl")
    encoder = joblib.load("labels.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")

    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image not found")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, img_size)
    enhanced = enhance_image(resized)

    hog_feat = extract_hog_features(enhanced)
    lbp_feat = extract_lbp_features(enhanced)
    sift_feat = extract_sift_features(enhanced)

    features = np.concatenate([hog_feat, lbp_feat, sift_feat]).reshape(1, -1)

    features = scaler.transform(features)
    features = pca.transform(features)

    prob = model.predict_proba(features)[0]
    pred = np.argmax(prob)

    name = encoder.inverse_transform([pred])[0]
    confidence = np.max(prob)

    # ✅ FINAL CLEAN OUTPUT (ONLY ONE RESULT)
    print(f"\n🔍 Final Prediction: {name} ({confidence*100:.2f}%)")

    # Display result
    cv2.putText(img, f"{name} ({confidence*100:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------- MAIN --------------------
if __name__ == "__main__":
    dataset_dir = r"C:\Users\akshi\OneDrive\Desktop\muzzle-recognition\Muzzletrainset"

    X, y = load_dataset(dataset_dir)
    train_model(X, y)

    test_image = r"C:\Users\akshi\OneDrive\Desktop\muzzle-recognition\Muzzletestset\cattle_1100\cattle_1100_DJI_0185.jpg"
    predict(test_image)