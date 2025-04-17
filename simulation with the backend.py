import cv2
import numpy as np
import joblib

# --- Preprocessing steps (must match training) ---
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def segment_leaf(image):
    image = cv2.resize(image, (512, 512))
    image = apply_clahe(image)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, 450, 450)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return image * mask[:, :, np.newaxis]

def extract_color_histogram(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_features = []
    for i in range(3):
        hist_rgb = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist_hsv = cv2.calcHist([image_hsv], [i], None, [256], [0, 256])
        hist_rgb = cv2.normalize(hist_rgb, hist_rgb).flatten()
        hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
        hist_features.extend(hist_rgb)
        hist_features.extend(hist_hsv)
    return np.array(hist_features)

def extract_glcm_features(image):
    from skimage.feature import graycomatrix, graycoprops
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0]
    ])

def extract_lbp_texture(image):
    from skimage.feature import local_binary_pattern
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    return hist.astype("float") / hist.sum()

def combine_features(image):
    return np.hstack([
        extract_color_histogram(image),
        extract_glcm_features(image),
        extract_lbp_texture(image)
    ])

# --- Classifier Function ---
def classifier(image_path):
    # Step 1: Dummy plant type classification
    plant_type = "pepper"  # later this will be replaced by a real classifier

    # Step 2: Load correct model components for that plant
    base_path = f"F:/Gradutation project/Machine learning models/last svm inshallah/{plant_type.capitalize()} last svm inshallah/"
    model = joblib.load(base_path + "svm_best_pepper_inshallah.pkl")
    pca = joblib.load(base_path + "pca_best_pepper_inshallah.pkl")
    scaler = joblib.load(base_path + "scaler_best_pepper_inshallah.pkl")
    label_map = joblib.load(base_path + "label_to_class.pkl")

    # Step 3: Load image and apply preprocessing
    image = cv2.imread(image_path)
    if image is None:
        return "Invalid image path."

    segmented = segment_leaf(image)
    features = combine_features(segmented)
    features_scaled = scaler.transform([features])
    features_pca = pca.transform(features_scaled)

    # Step 4: Predict class
    predicted_label = model.predict(features_pca)[0]
    class_name = label_map[predicted_label]

    return class_name

# ------------------------------
# All function definitions above
# ------------------------------

if __name__ == "__main__":
    result = classifier("F:/Gradutation project/Some Images/sample_leaf.jpg")
    print("This is a:", result)
