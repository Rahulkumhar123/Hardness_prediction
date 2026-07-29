import pandas as pd

# Read files
hardness_df = pd.read_csv(r"C:\Users\nitia\OneDrive\Desktop\SRIP_ISM\code_using_opencv_imgprocess\hardness.csv")
ils_df = pd.read_csv(r"C:\Users\nitia\OneDrive\Desktop\SRIP_ISM\code_using_opencv_imgprocess\ILS_results.csv")

# -----------------------------
# Step 1: Sort (important for consistency)
# -----------------------------
hardness_df = hardness_df.sort_values("image").reset_index(drop=True)
ils_df = ils_df.sort_values("image").reset_index(drop=True)

# -----------------------------
# Step 2: Rename images sequentially
# -----------------------------
hardness_df["image"] = [f"image{i+1}" for i in range(len(hardness_df))]
ils_df["image"] = [f"image{i+1}" for i in range(len(ils_df))]

# -----------------------------
# Step 3: Merge on new names
# -----------------------------
training_df = pd.merge(ils_df, hardness_df, on="image")

# -----------------------------
# Step 4: Save dataset
# -----------------------------
output_path = r"C:\Users\nitia\OneDrive\Desktop\SRIP_ISM\code_using_opencv_imgprocess\model\training_dataset.csv"
training_df.to_csv(output_path, index=False)

print("Dataset created successfully!")
print(training_df.head())
print("\nShape:", training_df.shape)