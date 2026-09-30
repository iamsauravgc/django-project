from django import forms


class EmotionForm(forms.Form):
    text = forms.CharField(
        label="Your text",
        max_length=2000,
        min_length=3,
        strip=True,
        widget=forms.Textarea(
            attrs={
                "rows": 5,
                "placeholder": "Type or paste anything: a journal entry, a review, a message...",
                "autofocus": True,
            }
        ),
        error_messages={
            "required": "Paste some text to analyze.",
            "min_length": "Give me at least 3 characters to work with.",
        },
    )
