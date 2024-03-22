"""This code provides a structured framework for evaluating and selecting models by 
first preparing the data, splitting it into training and testing sets, and defining 
evaluation metrics. It then performs hyperparameter tuning and model training using grid
 search and cross-validation techniques. The results include the identification of the best 
 parameters for each model and detailed evaluation reports, enabling informed decision-making
   in selecting the most effective algorithm and hyperparameters for the given dataset. 
   Additionally, it exports the plots as PNG files for further analysis. Overall, it offers
     a systematic approach to model evaluation and selection, ensuring robust performance 
     assessment and optimization."""




import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Function to split data into features and target variable
def split_to_X_y(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

# Read data and split it
chosen_data = pd.read_csv("C:/Users/NedyaIbrahim-AI23GBG/Documents/Github/Machinelearning_Nedya/Labb/asset/dataset_cleaned.csv")
X, y = split_to_X_y(chosen_data, "cardio")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Function to evaluate the model and save confusion matrix plot as PNG
def evaluate_model(model, test_X, test_y, title, filename):
    y_pred = model.predict(test_X)
    print(title)
    print(classification_report(test_y, y_pred))
    cm = confusion_matrix(test_y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.savefig(filename + ".png")  # Save confusion matrix plot as PNG
    plt.show()

# Pipelines for different algorithms
    
# Pipeline for KNN with StandardScaler
pipe_KNN = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# Pipeline for logistic regression with StandardScaler
pipe_log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=500, solver='saga'))
])

# Pipeline for Random Forest with StandardScaler
pipe_RFC = Pipeline([
    ("scaler", StandardScaler()),
    ("random_forest", RandomForestClassifier())
])

# Parameters for grid search for each algorithm
param_grid_log_reg = {
    'logistic__C': [0.01, 0.1, 1.0, 10, 100],
    'logistic__penalty': ['l1', 'l2'],
    'logistic__max_iter': [500, 1000, 5000, 10000, 20000],
    'logistic__tol': [1e-4, 1e-3, 1e-2], 
    'logistic__solver': ['liblinear', 'saga']  
}
param_grid_RFC = {'random_forest__n_estimators': list(range(20, 60))}
param_grid_KNN = {'knn__n_neighbors': [3, 5, 7, 9, 11]}

# Function for grid search and training for a given algorithm
def perform_grid_search_and_train(model_name, model_pipeline, param_grid, X_train, y_train, X_val, y_val):
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, verbose=1, scoring='recall')
    grid_search.fit(X_train, y_train)

    # Print best parameter combination
    print(f"Best parameters for {model_name}:", grid_search.best_params_)

    # Save the best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f"best_{model_name}_model.pkl")

    # Evaluate the best model on validation data
    evaluate_model(best_model, X_val, y_val, f"Classification report for {model_name}:", f"{model_name}_confusion_matrix")

    return grid_search

# Perform grid search and training for each algorithm
knn_grid_search = perform_grid_search_and_train("KNN", pipe_KNN, param_grid_KNN, X_train, y_train, X_test, y_test)
log_reg_grid_search = perform_grid_search_and_train("Logistic Regression", pipe_log_reg, param_grid_log_reg, X_train, y_train, X_test, y_test)
rfc_grid_search = perform_grid_search_and_train("Random Forest", pipe_RFC, param_grid_RFC, X_train, y_train, X_test, y_test)

# Print the results of the best models
print("Best KNN Model:")
evaluate_model(knn_grid_search.best_estimator_, X_test, y_test, "KNN", "knn_confusion_matrix")
print("Best Logistic Regression Model:")
evaluate_model(log_reg_grid_search.best_estimator_, X_test, y_test, "Logistic Regression", "logistic_regression_confusion_matrix")
print("Best Random Forest Model:")
evaluate_model(rfc_grid_search.best_estimator_, X_test, y_test, "Random Forest", "random_forest_confusion_matrix")