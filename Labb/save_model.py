"""The code performs several key tasks: it initially loads and prepares the dataset, 
then splits it into training and testing sets for model evaluation. Subsequently, a 
RandomForestClassifier model is trained on the training data, and its accuracy is assessed 
by predicting outcomes on the test set. Additionally, a subset of 100 rows is randomly selected
 from the dataset to create test samples, which are exported to a CSV file. Following this, the
   test samples are removed from the original dataset, and a second RandomForestClassifier model
     is trained on the remaining data. Finally, the best-performing model is saved to a .pkl 
     file with compression enabled for future use."""



import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data and split into features and target variable
chosen_data = chosen_data = pd.read_csv("C:/Users/NedyaIbrahim-AI23GBG/Documents/Github/Machinelearning_Nedya/Labb/asset/dataset_cleaned.csv")
X = chosen_data.drop(columns=["cardio"])
y = chosen_data["cardio"]

# Sample 100 rows randomly from the dataset
test_samples = chosen_data.sample(n=1000, random_state=42)

# Export the test samples to a CSV file
test_samples.to_csv("Labb/asset/test_samples.csv", index=False)

# Remove the test samples from the dataset
remaining_data = chosen_data.drop(test_samples.index)

# Split into train and test sets
X_remaining = remaining_data.drop(columns=["cardio"])
y_remaining = remaining_data["cardio"]
X_train, X_test, y_train, y_test = train_test_split(X_remaining, y_remaining, test_size=0.20, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the final model on training data
final_model = LogisticRegression(max_iter=1000)  # Increase max_iter
final_model.fit(X_train_scaled, y_train)

# Evaluate the final model on test data
y_pred = final_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the final model on the test set: {accuracy:.4f}")

# Save the trained model to a .pkl file
joblib.dump(final_model, "Labb/asset/trained_model.pkl", compress=True)