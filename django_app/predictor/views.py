from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count
from django.shortcuts import render

from .forms import EmotionForm
from .models import EMOTION_COLORS, EMOTION_LABELS, Prediction
from .utils import ArtifactMissing, predict_emotion


@login_required
def predict_view(request):
    """Analyze text on GET/POST and persist every successful run."""
    result = None

    if request.method == "POST":
        form = EmotionForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data["text"]
            try:
                result = predict_emotion(text)
            except ArtifactMissing as exc:
                messages.error(request, str(exc))
            else:
                Prediction.objects.create(
                    user=request.user,
                    text=text,
                    predicted_emotion=result["predicted"],
                    confidence=result["confidence"],
                    all_scores=result["scores"],
                )
    else:
        form = EmotionForm()

    return render(
        request,
        "predictor/predict.html",
        {"form": form, "result": result},
    )


@login_required
def history_view(request):
    """The signed-in user's own predictions, nobody else's."""
    predictions = Prediction.objects.filter(user=request.user)
    counts = (
        Prediction.objects.filter(user=request.user)
        .values("predicted_emotion")
        .annotate(n=Count("id"))
        .order_by("-n")
    )
    summary = [
        {
            "emotion": row["predicted_emotion"],
            "label": EMOTION_LABELS[row["predicted_emotion"]],
            "color": EMOTION_COLORS[row["predicted_emotion"]],
            "n": row["n"],
        }
        for row in counts
    ]

    return render(
        request,
        "predictor/history.html",
        {
            "predictions": predictions,
            "summary": summary,
            "total": sum(row["n"] for row in summary),
        },
    )
