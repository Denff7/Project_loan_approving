from django.db import models
from django.contrib.auth.models import User


class LoanPrediction(models.Model):
    """
        Stores historical loan approval predictions for users.

        This model keeps a record of the input parameters (demographic and financial
        data) provided by a user during a loan prediction request, along with the
        resulting probability of approval calculated by the machine learning model.

        Attributes:
            user (ForeignKey): The user who made the prediction request.
            married (str): Marital status of the applicant ('Yes' or 'No').
            dependents (str): Number of dependents.
            education (str): Educational background of the applicant.
            self_employed (str): Employment status ('Yes' or 'No').
            property_area (str): The area where the property is located.
            applicant_income (float): The primary applicant's income.
            coapplicant_income (float): The co-applicant's income.
            loan_amount (float): The requested loan amount.
            loan_amount_term (float): The term of the loan in months.
            credit_history (float): Record of past credit history (typically 1.0 or 0.0).
            probability (float): The calculated chance of loan approval (0-100%).
            created_at (datetime): The exact date and time the request was made.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    married = models.CharField(max_length=3, choices=[('Yes', 'Так'), ('No', 'Ні')])
    dependents = models.CharField(max_length=5)
    education = models.CharField(max_length=20)
    self_employed = models.CharField(max_length=3)
    property_area = models.CharField(max_length=20)
    applicant_income = models.FloatField()
    coapplicant_income = models.FloatField()
    loan_amount = models.FloatField()
    loan_amount_term = models.FloatField()
    credit_history = models.FloatField()
    probability = models.FloatField(help_text="Ймовірність схвалення у відсотках")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Дата запиту")

    def __str__(self):
        """
                Returns a string representation of the loan prediction instance.
        """
        return f"Запит {self.user.username} - {self.probability}% ({self.created_at.strftime('%Y-%m-%d')})"