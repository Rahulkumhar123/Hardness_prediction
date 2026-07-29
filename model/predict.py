import os
import cv2
import pandas as pd
import joblib
import importlib.util
import sys

sys.path.append(
    r"C:\Users\nitia\OneDrive\Desktop\SRIP_ISM\code_using_opencv_imgprocess"
)


# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load(r"C:\Users\nitia\OneDrive\Desktop\SRIP_ISM\code_using_opencv_imgprocess\model\hardness_model.pkl")

# -----------------------------
# Load lamellar_spacing.py
# -----------------------------
from lamellar_spacing import calculate_interlamellar_spacing
# -----------------------------
# Test image folder
# -----------------------------
test_folder = r"C:\Users\nitia\OneDrive\Desktop\SRIP_ISM\code_using_opencv_imgprocess\test_images"

results = []

# -----------------------------
# Process all images
# -----------------------------
print("Test folder =", os.path.abspath(test_folder))
for image_name in sorted(os.listdir(test_folder)):
    print("Checking:", image_name)

    if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
        continue

    image_path = os.path.join(test_folder, image_name)
    

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Could not read {image_name}")
        continue

    # Calculate ILS


    ils_value = calculate_interlamellar_spacing(img)
    X_new = pd.DataFrame({"ILS_um": [ils_value]})

    predicted_hardness = model.predict(X_new)[0]

    results.append({
        "image": image_name,
        "ILS_um": ils_value,
        "Predicted_Hardness": predicted_hardness
    })

    

    # -----------------------------
    # Save results
    # -----------------------------
results_df = pd.DataFrame(results)
output_csv = "C:\Users\nitia\OneDrive\Desktop\SRIP_ISM\code_using_opencv_imgprocess\predicted_hardness_results.csv"
results_df.to_csv(output_csv, index=False)
print(f"\nResults saved to: {output_csv}")