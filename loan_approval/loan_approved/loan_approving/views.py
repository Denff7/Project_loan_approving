from django.shortcuts import render
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

"""
Request processing and business logic for the loan approval prediction application.

This module contains two main Django views:
1. index — displays the homepage.
2. loan_predict_view — processes form input, trains a machine learning model,
   and predicts the probability of loan approval.
"""


def index(request):
    """
    Render the main homepage of the application.

    Args:
        request (HttpRequest): The HTTP request object sent by the client.

    Returns:
        HttpResponse: The rendered HTML template for 'loan_approving/index.html'.
    """
    return render(request, "loan_approving/index.html")


def loan_predict_view(request):
    """
    Process user-submitted form data, train a Random Forest model using loan_data.csv,
    and predict the probability of loan approval.

    Workflow:
        1. Check if the request method is POST.
        2. Load a local CSV dataset containing previous loan applications.
        3. Clean missing values in important columns.
        4. Define categorical and numerical features.
        5. Build a preprocessing pipeline:
           - Encode categorical features (OneHotEncoder).
           - Scale numerical features (StandardScaler).
        6. Combine preprocessing with a RandomForestClassifier using a Pipeline.
        7. Train the model on the dataset.
        8. Collect user input from the HTML form.
        9. Create a new DataFrame with this input.
        10. Use the trained model to predict the probability of loan approval.
        11. Return the result to the loan_predict.html template.

    Args:
        request (HttpRequest): The HTTP request containing form data (POST).

    Returns:
        HttpResponse: Rendered HTML page showing the loan approval probability.
    """

    probability = None  # Placeholder for the prediction result

    if request.method == 'POST':
        # Path to the local dataset
        csv_path = r'D:\PyCharm projects\Loan_approved\loan_approval\loan_data.csv'
        users_df = pd.read_csv(csv_path)

        # Fill missing values in key columns
        users_df['Dependents'] = users_df['Dependents'].fillna(users_df['Dependents'].mode()[0])
        users_df['Self_Employed'] = users_df['Self_Employed'].fillna(users_df['Self_Employed'].mode()[0])
        users_df['Loan_Amount_Term'] = users_df['Loan_Amount_Term'].fillna(users_df['Loan_Amount_Term'].median())
        users_df['Credit_History'] = users_df['Credit_History'].fillna(users_df['Credit_History'].median())

        # Define feature categories
        categorical_features = ['Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                              'Credit_History']

        # Split features and target variable
        X = users_df[categorical_features + numerical_features]
        y = users_df['Loan_Status'].map({'Y': 1, 'N': 0})

        # Define preprocessing pipeline
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(), categorical_features),   # Encode categorical variables
            ('num', StandardScaler(), numerical_features)     # Scale numerical variables
        ])

        # Combine preprocessing and model into a single pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # Train the model
        pipeline.fit(X, y)

        # Collect user input from the form
        new_user_df = pd.DataFrame([{
            'Married': request.POST.get('Married'),
            'Dependents': request.POST.get('Dependents'),
            'Education': request.POST.get('Education'),
            'Self_Employed': request.POST.get('Self_Employed'),
            'Property_Area': request.POST.get('Property_Area'),
            'ApplicantIncome': float(request.POST.get('ApplicantIncome', 0)),
            'CoapplicantIncome': float(request.POST.get('CoapplicantIncome', 0)),
            'LoanAmount': float(request.POST.get('LoanAmount', 0)),
            'Loan_Amount_Term': float(request.POST.get('Loan_Amount_Term', 360)),
            'Credit_History': float(request.POST.get('Credit_History', 1))
        }])

        # Predict loan approval probability
        predicted = pipeline.predict_proba(new_user_df)
        probability = round(predicted[0][1] * 100, 0)  # Convert to percentage and round

    # Render the prediction result in the template
    return render(request, 'loan_approving/loan_predict.html', {'probability': probability})
