import pandas as pd

# Load the Iris dataset
df = pd.read_csv("iris (1).csv")

# Display the first five rows
print(df.head())

# Display the dataset size
print("\nDataset shape:")
print(df.shape)

# Count the flowers in each species
print("\nSpecies counts:")
print(df["species"].value_counts())
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize relationships between flower measurements
sns.pairplot(df, hue="species")
plt.show()
from sklearn.model_selection import train_test_split

# Separate input features and target
X = df.drop("species", axis=1)
y = df["species"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
from sklearn.preprocessing import StandardScaler

# Create the scaler
scaler = StandardScaler()

# Learn scaling values from training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same scaling to testing data
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")
print("Scaled training data shape:", X_train_scaled.shape)
print("Scaled testing data shape:", X_test_scaled.shape)
from sklearn.linear_model import LogisticRegression

# Create the Logistic Regression model
lr_model = LogisticRegression(max_iter=200)

# Train the model
lr_model.fit(X_train_scaled, y_train)

# Predict species for testing data
y_pred_lr = lr_model.predict(X_test_scaled)

print("\nLogistic Regression training completed!")
print("First 5 predictions:", y_pred_lr[:5])
print("First 5 actual values:", y_test.iloc[:5].tolist())
from sklearn.metrics import accuracy_score, classification_report

# Calculate accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print("\n--- Logistic Regression Evaluation ---")
print("Accuracy:", round(accuracy_lr, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred_lr)

# Display the confusion matrix
display = ConfusionMatrixDisplay(
confusion_matrix=cm,
display_labels=lr_model.classes_
)

display.plot(cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

print("\nConfusion Matrix:")
print(cm)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Predict the species
y_pred_dt = dt_model.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)

print("\n--- Decision Tree Evaluation ---")
print("Accuracy:", round(accuracy_dt, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))
# Compare both models
comparison = pd.DataFrame({
"Model": ["Logistic Regression", "Decision Tree"],
"Accuracy": [accuracy_lr, accuracy_dt]
})

print("\n--- Model Comparison ---")
print(comparison)

# Plot accuracy comparison
plt.figure(figsize=(7, 4))
sns.barplot(data=comparison, x="Model", y="Accuracy")
plt.title("Accuracy Comparison of Classification Models")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.show()
from sklearn.inspection import DecisionBoundaryDisplay

# Select two features for visualization
X_2d = df[["petal_length", "petal_width"]]
y_all = df["species"]

# Train a Decision Tree using only these two features
dt_2d = DecisionTreeClassifier(random_state=42)
dt_2d.fit(X_2d, y_all)

# Plot the decision boundaries
plt.figure(figsize=(8, 5))

DecisionBoundaryDisplay.from_estimator(
dt_2d,
X_2d,
response_method="predict",
cmap="Pastel1",
alpha=0.6,
xlabel="Petal Length",
ylabel="Petal Width",
ax=plt.gca()
)

sns.scatterplot(
data=df,
x="petal_length",
y="petal_width",
hue="species",
edgecolor="black"
)

plt.title("Decision Boundaries - Decision Tree")
plt.show()

print("\n========== FINAL PROJECT SUMMARY ==========")
print("Dataset: Iris")
print("Number of samples:", len(df))
print("Number of classes:", df["species"].nunique())
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Logistic Regression Accuracy:", round(accuracy_lr * 100, 2), "%")
print("Decision Tree Accuracy:", round(accuracy_dt * 100, 2), "%")

if accuracy_lr > accuracy_dt:
    print("Better model on this test split: Logistic Regression")
elif accuracy_dt > accuracy_lr:
    print("Better model on this test split: Decision Tree")
else:
    print("Both models have equal accuracy on this test split.")

print("===========================================")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate metrics for both models
comparison = pd.DataFrame({
"Model": ["Logistic Regression", "Decision Tree"],
"Accuracy": [
accuracy_score(y_test, y_pred_lr),
accuracy_score(y_test, y_pred_dt)
],
"Precision": [
precision_score(y_test, y_pred_lr, average="weighted", zero_division=0),
precision_score(y_test, y_pred_dt, average="weighted", zero_division=0)
],
"Recall": [
recall_score(y_test, y_pred_lr, average="weighted", zero_division=0),
recall_score(y_test, y_pred_dt, average="weighted", zero_division=0)
],
"F1-Score": [
f1_score(y_test, y_pred_lr, average="weighted", zero_division=0),
f1_score(y_test, y_pred_dt, average="weighted", zero_division=0)
]
})

print("\n========== FINAL MODEL COMPARISON ==========")
print(comparison.round(4).to_string(index=False))

