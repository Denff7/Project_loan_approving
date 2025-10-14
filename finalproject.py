"""
Loan Approval Prediction — Data Analysis and Model Training Script

This script performs data preprocessing, visualization, and machine learning
model training to predict loan approval status using the Random Forest algorithm.

Main stages:
1. Data loading and preprocessing (handling missing values, encoding categorical features)
2. Exploratory Data Analysis (EDA) with visualizations
3. Model training and evaluation using Random Forest within a pipeline
4. Hyperparameter optimization using GridSearchCV
5. Prediction for a new user input
6. Visualization of feature importance and data distributions
"""

# === Import Libraries ===
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


# === 1. Load Dataset ===
users_df = pd.read_csv('loan_data.csv')


# === 2. Handle Missing Values ===
# Fill missing values in categorical columns using mode
mode_dependents = users_df['Dependents'].mode()[0]
mode_self_employed = users_df['Self_Employed'].mode()[0]
users_df['Dependents'] = users_df['Dependents'].fillna(mode_dependents)
users_df['Self_Employed'] = users_df['Self_Employed'].fillna(mode_self_employed)

# Fill missing values in numerical columns using median
median_loan_amount_term = users_df['Loan_Amount_Term'].median()
median_credit_history = users_df['Credit_History'].median()
users_df['Loan_Amount_Term'] = users_df['Loan_Amount_Term'].fillna(median_loan_amount_term)
users_df['Credit_History'] = users_df['Credit_History'].fillna(median_credit_history)


# === 3. Correlation Analysis (Initial Heatmap) ===
numeric = users_df.select_dtypes(include=[np.number])
corr = numeric.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap — Original Data")
plt.show()


# === 4. One-Hot Encoding for Categorical Features ===
data = users_df[['Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']]
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(data)

encoder_users = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=users_df.index)

# Combine encoded data with original dataset
encoder_users = pd.concat([users_df, encoder_users], axis=1).drop(
    ['Loan_ID', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'],
    axis=1
)

# Define target and features
y = encoder_users['Loan_Status'].map({'Y': 1, 'N': 0})
X = encoder_users.drop(['Gender', 'Loan_Status'], axis=1)


# === 5. Correlation Heatmap (After Encoding) ===
numeric = X.select_dtypes(include=[np.number])
corr = numeric.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap — Encoded Features")
plt.show()


# === 6. Split Data into Training and Test Sets ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# === 7. Define Pipeline with Scaling and Random Forest ===
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", f'{accuracy * 100:.2f}%')


# === 8. Cross-Validation ===
scorepipeline = cross_val_score(pipeline, X, y, cv=20, scoring='accuracy')

print("Accuracy scores for each fold of Random Forest (%):")
print([f"{score:.2f}%" for score in scorepipeline * 100])
print(f"Mean cross-validation accuracy: {scorepipeline.mean() * 100:.2f}%\n")


# === 9. Hyperparameter Tuning with GridSearchCV ===
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7, None],
}

grid_rf = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_rf.fit(X_train, y_train)

print("Best settings for Random Forest:", grid_rf.best_params_)
print(f"Best accuracy on training data: {grid_rf.best_score_ * 100:.2f}%")


# === 10. Predict for a New User ===
new_user = [[4583, 1508, 128, 360, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]]
columns = X_train.columns
new_user_df = pd.DataFrame(new_user, columns=columns)

predicted_proba = pipeline.predict_proba(new_user_df)
print(f"Predicted probability of loan approval: {predicted_proba[0][1]*100:.2f}%")


# === 11. Feature Importance Visualization ===
importances = pipeline.named_steps['classifier'].feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()


# === 12. Additional Data Visualizations ===

# Boxplot: Applicant income vs loan status
plt.figure(figsize=(10, 6))
sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=users_df)
plt.title("Applicant Income Distribution by Loan Status")
plt.xlabel("Loan Status (0 = Denied, 1 = Approved)")
plt.ylabel("Applicant Income")
plt.show()

# Heatmap of numerical correlations
plt.figure(figsize=(10, 6))
corr = users_df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Between Numerical Features")
plt.show()

# Histogram: Applicant Income
plt.figure(figsize=(10, 6))
plt.hist(users_df['ApplicantIncome'], bins=100)
plt.title("Distribution of Applicant Income")
plt.xlabel("Income")
plt.ylabel("Number of Applicants")
plt.show()

# Histogram: Loan Amount
plt.figure(figsize=(10, 6))
plt.hist(users_df["LoanAmount"], bins=10, edgecolor='black')
plt.title("Distribution of Loan Amounts")
plt.xlabel("Loan Amount ($)")
plt.ylabel("Number of Applications")
plt.show()
