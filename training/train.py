MODEL_NAME = "mhand_gesture_model.pkl"
TRAINING_DATA_PATH = "training/mouin_data.csv"

# pip install pandas scikit-learn joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd

df=pd.read_csv(TRAINING_DATA_PATH)
# 4) Split into train/test
X = df.drop("label", axis=1).values  # shape (N,63)
y = df["label"].values               # shape (N,)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5) Configure and train the MLP classifier
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42,
    verbose=True
)
model.fit(X_train, y_train)

# 6) Evaluate on test set
y_pred = model.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# 7) Save the trained model for later inference
joblib.dump(model, MODEL_NAME)
print("\nModel saved to hand_gesture_model.pkl")
