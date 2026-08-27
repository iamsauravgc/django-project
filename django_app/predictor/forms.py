"""
Epic 5 — Forms boilerplate (E5-T2)
"""
from django import forms

class EmotionForm(forms.Form):
    # TODO: widget=Textarea, max_length=2000, placeholder="Type or paste..."
    text = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 4, "placeholder": "Type or paste text..."}),
        label="Your text",
        max_length=2000,
    )
