"""This code serves the following purpose: to read in the test samples and the trained model
 from the CSV files "test_samples.csv" and "trained_model.pkl" respectively. Subsequently,
    it performs predictions on the test samples using the trained model and creates a DataFrame
    containing the predictions along with the probabilities for each class. Finally, these
       predictions are exported to a CSV file named "predictions.csv". The results in this file 
       display the probability for each class and the predicted class for each individual 
       example"""


import pandas as pd
import joblib

# Load the test samples and the trained model
test_samples = pd.read_csv("Labb/asset/test_samples.csv")
trained_model = joblib.load("Labb/asset/trained_model.pkl")

# Remove the "cardio" column from the test samples
test_samples = test_samples.drop(columns=['cardio'], errors='ignore')

# Make predictions on the test samples
predictions = trained_model.predict(test_samples)

# Get probabilities for each class
probabilities = trained_model.predict_proba(test_samples)

# Create a DataFrame for predictions
prediction_df = pd.DataFrame({
    "probability class 0": probabilities[:, 0],
    "probability class 1": probabilities[:, 1],
    "prediction": predictions
})

# Export predictions to a CSV file in the "assets" directory
prediction_df.to_csv("Labb/asset/predictions.csv", index=False)

# Läs in filen med förutsägelser från "assets" mappen
predictions = pd.read_csv("Labb/asset/predictions.csv")

# Visa de första några rader av data
print(predictions.head())