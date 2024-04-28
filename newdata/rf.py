from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from src.utilities import create_data
import pandas as pd

X, y = create_data(folder="../dataset")
y = y.flatten().astype(int)

"""
df = pd.read_json("jaccard.json")
df.drop(["DRUG_ID1", "DRUG_ID2"], inplace=True, axis=1)

# Define features (X) and target variable (y)
X = df.drop('INTERACTION', axis=1)  # Features
y = df['INTERACTION']  # Target variable
"""

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y_test, y_pred)
print("Matthews Correlation Coefficient:", mcc)

# Print classification report
print(classification_report(y_test, y_pred))