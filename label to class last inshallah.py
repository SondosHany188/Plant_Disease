import os
import joblib

folder_path = "F:/Gradutation project/Dataset/plant village/Dataset divided/potato"

# Get classes in their natural filesystem order (not sorted)
classes = os.listdir(folder_path)

# Create label-to-class mapping based on the original order
label_to_class = {label: class_name for label, class_name in enumerate(classes)}

# Print the mapping
print("Label to Class Mapping:")
for label, class_name in label_to_class.items():
    print(f"{label}: {class_name}")

# Save the mapping
joblib.dump(label_to_class, "F:/Gradutation project/Machine learning models/last svm inshallah/Potato last svm inshallah/Potato_label_to_class.pkl")
