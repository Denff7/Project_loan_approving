import pandas as pd
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from .models import LoanPrediction


def index(request):
    """
        Render the home page based on user authentication status.

        Args:
            request (HttpRequest): The incoming HTTP request.

        Returns:
            HttpResponse: Renders 'loan_approving/index.html' if the user is
            authenticated, otherwise renders 'loan_approving/landing.html'.
    """
    if not request.user.is_authenticated:
        return render(request, "loan_approving/landing.html")
    return render(request, "loan_approving/index.html")


def signup_view(request):
    """
        Handle user registration.

        Processes the UserCreationForm on POST requests to create a new user account.
        Upon successful validation, automatically logs the new user in and redirects
        them to the index page. Renders an empty form on GET requests.

        Args:
            request (HttpRequest): The incoming HTTP request.

        Returns:
            HttpResponse: A redirect to the 'index' view on successful registration,
            or a rendered 'registration/signup.html' template containing the form.
    """
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('index')
    else:
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})


@login_required
def loan_predict_view(request):
    """
        Calculate and display the probability of loan approval based on user input.

        On a POST request, this view loads historical CSV data, trains a Random
        Forest Classifier with preprocessing (handling missing values, encoding,
        and scaling), and predicts the approval probability using the submitted form
        data. Finally, it saves the input parameters and prediction result to the
        database for the current user.

        Args:
            request (HttpRequest): The incoming HTTP request containing form data.

        Returns:
            HttpResponse: Renders the 'loan_approving/loan_predict.html' template,
            passing the calculated 'probability' integer to the context.
    """
    probability = None
    if request.method == 'POST':
        csv_path = r'D:\PyCharm projects\Loan_approved\loan_data.csv'
        users_df = pd.read_csv(csv_path)
        users_df['Dependents'] = users_df['Dependents'].fillna(users_df['Dependents'].mode()[0])
        users_df['Self_Employed'] = users_df['Self_Employed'].fillna(users_df['Self_Employed'].mode()[0])
        users_df['Loan_Amount_Term'] = users_df['Loan_Amount_Term'].fillna(users_df['Loan_Amount_Term'].median())
        users_df['Credit_History'] = users_df['Credit_History'].fillna(users_df['Credit_History'].median())
        categorical_features = ['Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                              'Credit_History']

        X = users_df[categorical_features + numerical_features]
        y = users_df['Loan_Status'].map({'Y': 1, 'N': 0})

        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ])
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        pipeline.fit(X, y)
        raw_data = {
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
        }

        new_user_df = pd.DataFrame([raw_data])
        predicted = pipeline.predict_proba(new_user_df)
        probability = round(predicted[0][1] * 100, 0)
        LoanPrediction.objects.create(
            user=request.user,
            married=raw_data['Married'],
            dependents=raw_data['Dependents'],
            education=raw_data['Education'],
            self_employed=raw_data['Self_Employed'],
            property_area=raw_data['Property_Area'],
            applicant_income=raw_data['ApplicantIncome'],
            coapplicant_income=raw_data['CoapplicantIncome'],
            loan_amount=raw_data['LoanAmount'],
            loan_amount_term=raw_data['Loan_Amount_Term'],
            credit_history=raw_data['Credit_History'],
            probability=probability
        )
    return render(request, 'loan_approving/loan_predict.html', {'probability': probability})

@login_required
def history_view(request):
    """
        Display the past loan predictions for the currently logged-in user.

        Retrieves all LoanPrediction records associated with the user making the
        request and orders them chronologically from newest to oldest.

        Args:
            request (HttpRequest): The incoming HTTP request.

        Returns:
            HttpResponse: Renders the 'loan_approving/history.html' template,
            passing the user's prediction 'history' query list to the context.
    """
    user_history = LoanPrediction.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'loan_approving/history.html', {'history': user_history})