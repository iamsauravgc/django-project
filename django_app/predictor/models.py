"""
Epic 5 — Models boilerplate (E5-T6)
"""
from django.db import models
# TODO: from django.contrib.auth.models import User

# Boilerplate choices per CLAUDE.md:196 + Q5 (7 emotions)
EMOTION_CHOICES = [
    ("joy", "Joy"),
    ("anger", "Anger"),
    ("fear", "Fear"),
    ("sadness", "Sadness"),
    ("surprise", "Surprise"),
    ("disgust", "Disgust"),
    ("neutral", "Neutral"),
]

# TODO: class Prediction(models.Model):
#     user = ForeignKey(User, on_delete=CASCADE)
#     text = TextField()
#     predicted_emotion = CharField(choices=EMOTION_CHOICES)
#     confidence = FloatField()
#     all_scores = JSONField(default=dict)
#     created_at = DateTimeField(auto_now_add=True)
#     class Meta: ordering = ["-created_at"]

# Minimal stub to keep migrate runnable
from django.contrib.auth.models import User

class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="predictions")
    text = models.TextField()
    predicted_emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES)
    confidence = models.FloatField(default=0.0)
    all_scores = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user}: {self.predicted_emotion}"
