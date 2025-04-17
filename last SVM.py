import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Function to apply CLAHE
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Function to segment leaf
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

# Extract Color Histogram
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

# Extract GLCM Features
def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0]
    ])

# Extract LBP Texture
def extract_lbp_texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    return hist.astype("float") / hist.sum()

# Combine all features
def combine_features(image):
    return np.hstack([
        extract_color_histogram(image),
        extract_glcm_features(image),
        extract_lbp_texture(image)
    ])

# Load dataset
def load_dataset(folder_path):
    X, y = [], []
    classes = os.listdir(folder_path)
    label_to_class = {}  # <-- Added this to map numeric labels to class names
    for label, class_name in enumerate(classes):
        label_to_class[label] = class_name  # <-- Save mapping
        class_path = os.path.join(folder_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                segmented_image = segment_leaf(image)
                X.append(combine_features(segmented_image))
                y.append(label)
    print("Label to class mapping:", label_to_class)  # <-- Optional: print the mapping
    return np.array(X), np.array(y)

# Load dataset path
folder_path = ""
X, y = load_dataset(folder_path)

# Train-Validation-Test split (70%-20%-10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

# Feature scaling
scaler = StandardScaler()

# Fit scaler on training data and transform it
X_train = scaler.fit_transform(X_train)

# Transform validation and test data using the fitted scaler
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Dimensionality reduction (PCA)
pca = PCA(n_components=100)

# Fit PCA on training data and transform it
X_train = pca.fit_transform(X_train)

# Transform validation and test data using the fitted PCA
X_val = pca.transform(X_val)
X_test = pca.transform(X_test)

# Train SVM with GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1], 'kernel': ['rbf']}
svm = GridSearchCV(SVC(), param_grid, cv=5)
svm.fit(X_train, y_train)

# Save the trained model & PCA
joblib.dump(svm,"")
joblib.dump(pca, "")
joblib.dump(scaler, "")
# Evaluate on Training Set
y_train_pred = svm.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))

# Evaluate on Validation Set
y_val_pred = svm.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# Confusion Matrix for Validation Set
conf_matrix_val = confusion_matrix(y_val, y_val_pred)
disp_val = ConfusionMatrixDisplay(conf_matrix_val, display_labels=np.unique(y))
disp_val.plot(cmap=plt.cm.Blues)
plt.title("Validation Confusion Matrix")
plt.show()

# Reload model & test
svm_loaded = joblib.load("")
pca_loaded = joblib.load("")
scaler_loaded = joblib.load("")

# Evaluate on Test Set
y_test_pred = svm_loaded.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# Confusion Matrix for Test Set
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
disp_test = ConfusionMatrixDisplay(conf_matrix_test, display_labels=np.unique(y))
disp_test.plot(cmap=plt.cm.Blues)
plt.title("Test Confusion Matrix")
plt.show()
