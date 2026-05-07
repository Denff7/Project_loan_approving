from django.db import models
from django.contrib.auth.models import User


class LoanPrediction(models.Model):
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
        return f"Запит {self.user.username} - {self.probability}% ({self.created_at.strftime('%Y-%m-%d')})"