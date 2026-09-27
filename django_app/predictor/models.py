"""
Prediction records — one row per analysis, owned by the user who ran it.
"""
from django.contrib.auth.models import User
from django.db import models

EMOTION_CHOICES = [
    ("joy", "Joy"),
    ("anger", "Anger"),
    ("fear", "Fear"),
    ("sadness", "Sadness"),
    ("surprise", "Surprise"),
    ("disgust", "Disgust"),
    ("neutral", "Neutral"),
]

EMOTION_LABELS = dict(EMOTION_CHOICES)

# Mirrors ml/preprocess.EMOTION_COLORS (kept here so templates and the
# inference layer can pick colours without importing the whole ML stack).
EMOTION_COLORS = {
    "joy": "#FFD93D",
    "anger": "#FF6B6B",
    "fear": "#9D65C9",
    "sadness": "#6EC1E4",
    "surprise": "#FF9F45",
    "disgust": "#6BCB77",
    "neutral": "#C0C0C0",
}


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

    @property
    def color(self) -> str:
        return EMOTION_COLORS.get(self.predicted_emotion, "#C0C0C0")

    @property
    def confidence_pct(self) -> str:
        return f"{self.confidence * 100:.1f}%"
