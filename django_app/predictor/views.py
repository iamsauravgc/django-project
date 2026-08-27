"""
Epic 4 + 5 — Views boilerplate
"""
from django.shortcuts import render
# TODO: from django.contrib.auth.decorators import login_required
# TODO: from .forms import EmotionForm; from .models import Prediction; from .utils import predict_emotion

# TODO: @login_required
def predict_view(request):
    """TODO E5-T2..T6:
    - GET: render form
    - POST: form.is_valid() -> predict_emotion(text) -> Prediction.objects.create(user=request.user, ...)
    - sort scores, render result.html with color bars per CLAUDE.md:205
    """
    # stub: just render form
    return render(request, "predictor/predict.html", {})

# TODO: @login_required
def history_view(request):
    """TODO E5-T7: Prediction.objects.filter(user=request.user) — isolated per user"""
    return render(request, "predictor/history.html", {})
