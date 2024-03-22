"""The code separates both datasets into feature and target variables, then splits 
them into training, validation, and test sets. Afterward, it scales the data using 
both standardized and normalized scales for categorical and non-categorical datasets, 
 generating scaled datasets for machine learning models.In summary, the code handles 
 data preparation, splitting, and scaling to create training, validation, and test sets
   for machine learning models.By performing these steps, the code ensures that the data
     is properly processed, divided, and scaled, making it ready for training, validating, 
     and evaluating machine learning models"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Function to separate feature and target variables
def feature_target(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

# Function to split data into train, validation, and test sets
def split_data(X, y, test_val_size):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_val_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Function to scale data
def scale_data(X_train, X_val, scaler):
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled

# Read dataset
df = pd.read_csv(r"C:\Users\NedyaIbrahim-AI23GBG\Documents\Github\Machinelearning-Nedya-Ibrahim\Labb\asset\dataset_cleaned.csv")


# Create dataset with categorical features (df_cat)
df_categorical = df.drop(["ap_hi", "ap_lo", "height", "weight", "bmi"], axis=1)
df_categorical = pd.get_dummies(df_categorical, columns=["bmi_category", "bp_category", "gender"], prefix=["bmi_cat", "bp_cat", "sex"])

# Create dataset with non-categorical features (df_non_categorical)
df_non_categorical = df.drop(["bmi_category", "bp_category", "height", "weight"], axis=1)
df_non_categorical = pd.get_dummies(df_non_categorical, columns=["gender"], prefix=["sex"])

# Separate feature and target variables for categorical and non-categorical datasets
X_cat, y_cat = feature_target(df_categorical, target="cardio")
X_raw, y_raw = feature_target(df_non_categorical, target="cardio")

# Split data into train, validation, and test sets for both datasets
test_val_size = 0.2
X_cat_train, X_cat_val, X_cat_test, y_cat_train, y_cat_val, y_cat_test = split_data(X_cat, y_cat, test_val_size)
X_raw_train, X_raw_val, X_raw_test, y_raw_train, y_raw_val, y_raw_test = split_data(X_raw, y_raw, test_val_size)

# Scale X data: standardized for categorical and non-categorical datasets
scaler_std = StandardScaler()
X_cat_train_scaled_std, X_cat_val_scaled_std = scale_data(X_cat_train, X_cat_val, scaler_std)
X_raw_train_scaled_std, X_raw_val_scaled_std = scale_data(X_raw_train, X_raw_val, scaler_std)

# Scale X data: normalized for categorical and non-categorical datasets
scaler_minmax = MinMaxScaler()
X_cat_train_scaled_minmax, X_cat_val_scaled_minmax = scale_data(X_cat_train, X_cat_val, scaler_minmax)
X_raw_train_scaled_minmax, X_raw_val_scaled_minmax = scale_data(X_raw_train, X_raw_val, scaler_minmax)



# Plot heatmap for df_categorical
plt.figure(figsize=(10, 8))
sns.heatmap(df_categorical.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap - df_categorical")
plt.show()

plt.savefig("maps/correlation_heatmap_df_categorical.png")
plt.close()

# Plot heatmap for df_non_categorical
plt.figure(figsize=(10, 8))
sns.heatmap(df_non_categorical.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap - df_non_categorical")
plt.show()

plt.savefig("maps/correlation_heatmap_df_non_categorical.png")
plt.close()

# Print correlation matrices
print("Correlation Heatmap - df_categorical")
print(df_categorical.corr())

print("\nCorrelation Heatmap - df_non_categorical")
print(df_non_categorical.corr())