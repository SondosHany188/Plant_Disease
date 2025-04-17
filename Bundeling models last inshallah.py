import joblib

# Load the individual components
svm = joblib.load("F:/Gradutation project/Machine learning models/last svm inshallah/Potato last svm inshallah/svm_best_potato_inshallah.pkl")
pca = joblib.load("F:/Gradutation project/Machine learning models/last svm inshallah/Potato last svm inshallah/pca_best_potato_inshallah.pkl")
scaler = joblib.load("F:/Gradutation project/Machine learning models/last svm inshallah/Potato last svm inshallah/scaler_best_potato_inshallah.pkl")
label_to_class = joblib.load("F:/Gradutation project/Machine learning models/last svm inshallah/Potato last svm inshallah/Potato_label_to_class.pkl")

bundle = {
    "model": svm,
    "scaler": scaler,
    "pca": pca,
    "label_map": label_to_class
}

joblib.dump(bundle, "F:/Gradutation project/Machine learning models/last svm inshallah/Potato svm bundled inshallah/Potato_bundle.pkl")
