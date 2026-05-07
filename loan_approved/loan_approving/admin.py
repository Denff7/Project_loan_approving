from django.contrib import admin
from .models import LoanPrediction


@admin.register(LoanPrediction)
class LoanPredictionAdmin(admin.ModelAdmin):
    """
    Admin interface configuration for the LoanPrediction model.
    """
    # Columns to display in the admin list view
    list_display = ('user', 'applicant_income', 'loan_amount', 'probability', 'created_at')

    # Add a sidebar filter (by user, credit history, and education)
    list_filter = ('user', 'credit_history', 'education')

    # Add a search bar to filter records by the user's username
    search_fields = ('user__username',)

    # Default sorting from newest to oldest
    ordering = ('-created_at',)

