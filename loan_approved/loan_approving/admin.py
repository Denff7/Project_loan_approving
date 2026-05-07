from django.contrib import admin
from .models import LoanPrediction


@admin.register(LoanPrediction)
class LoanPredictionAdmin(admin.ModelAdmin):
    # Які колонки показувати в таблиці
    list_display = ('user', 'applicant_income', 'loan_amount', 'probability', 'created_at')

    # Додаємо панель фільтрів збоку (по користувачу та кредитній історії)
    list_filter = ('user', 'credit_history', 'education')

    # Додаємо рядок пошуку по імені користувача
    search_fields = ('user__username',)

    # Сортування від найновіших до найстаріших
    ordering = ('-created_at',)
# Register your models here.
