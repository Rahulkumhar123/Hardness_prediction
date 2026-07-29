
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\nitia\OneDrive\Desktop\SRIP_ISM\code_using_opencv_imgprocess\model\training_dataset.csv")

# Features and target
X = df[["ILS_um"]]
y = df["Hardness"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

save_path = r"C:\Users\nitia\OneDrive\Desktop\SRIP_ISM\code_using_opencv_imgprocess\model\hardness_model.pkl"

joblib.dump(model, save_path)

print("Model saved at:", save_path)