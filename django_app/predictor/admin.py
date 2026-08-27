from django.contrib import admin
from .models import Prediction

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("user", "predicted_emotion", "confidence", "created_at", "short_text")
    list_filter = ("predicted_emotion", "created_at")
    search_fields = ("text", "user__username")

    def short_text(self, obj):
        return obj.text[:50]
