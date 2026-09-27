from io import StringIO

from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse

from .models import EMOTION_CHOICES, Prediction
from .utils import load_artifacts, predict_emotion

LABELS = [label for label, _ in EMOTION_CHOICES]


class ArtifactTests(TestCase):
    def test_artifacts_load_with_seven_classes(self):
        model, vectorizer = load_artifacts()
        self.assertEqual(sorted(model.classes_), sorted(LABELS))
        self.assertTrue(hasattr(vectorizer, "transform"))

    def test_predict_emotion_returns_ranked_scores(self):
        result = predict_emotion("I am so happy, this is the best day ever!")

        self.assertIn(result["predicted"], LABELS)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

        scores = result["scores"]
        self.assertEqual(len(scores), len(LABELS))
        values = list(scores.values())
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertAlmostEqual(sum(values), 1.0, places=5)

        self.assertEqual(len(result["score_list"]), len(LABELS))
        self.assertTrue(result["predicted_color"].startswith("#"))

    def test_predict_emotion_handles_stopword_only_input(self):
        result = predict_emotion("the and or but")
        self.assertIn(result["predicted"], LABELS)
        self.assertEqual(len(result["scores"]), len(LABELS))


class PredictViewTests(TestCase):
    def setUp(self):
        User.objects.create_user("dan", password="Passw0rd!xyz")
        self.client.login(username="dan", password="Passw0rd!xyz")

    def test_get_renders_form(self):
        response = self.client.get(reverse("predict"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Your text")

    def test_post_saves_prediction_for_logged_in_user(self):
        before = Prediction.objects.count()
        response = self.client.post(
            reverse("predict"),
            {"text": "I absolutely love this, it made my whole day!"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Predicted emotion")

        self.assertEqual(Prediction.objects.count(), before + 1)
        prediction = Prediction.objects.get()
        self.assertEqual(prediction.user.username, "dan")
        self.assertIn(prediction.predicted_emotion, LABELS)
        self.assertEqual(len(prediction.all_scores), 7)
        self.assertAlmostEqual(sum(prediction.all_scores.values()), 1.0, places=5)
        self.assertGreaterEqual(prediction.confidence, 0.0)
        self.assertLessEqual(prediction.confidence, 1.0)

    def test_post_rejects_too_short_text(self):
        before = Prediction.objects.count()
        response = self.client.post(reverse("predict"), {"text": "a"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "errorlist")
        self.assertEqual(Prediction.objects.count(), before)

    def test_post_requires_login(self):
        self.client.logout()
        response = self.client.post(
            reverse("predict"), {"text": "This should not be saved at all."}
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Prediction.objects.count(), 0)


class HistoryViewTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("owner", password="Passw0rd!xyz")
        self.other = User.objects.create_user("other", password="Passw0rd!xyz")

        Prediction.objects.create(
            user=self.owner,
            text="My own joyful line about the weekend",
            predicted_emotion="joy",
            confidence=0.8,
            all_scores={"joy": 0.8},
        )
        Prediction.objects.create(
            user=self.other,
            text="Someone else private sad sentence",
            predicted_emotion="sadness",
            confidence=0.7,
            all_scores={"sadness": 0.7},
        )

    def test_history_shows_only_own_rows(self):
        self.client.login(username="owner", password="Passw0rd!xyz")
        response = self.client.get(reverse("history"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "My own joyful line about the weekend")
        self.assertNotContains(response, "Someone else private sad sentence")
        self.assertEqual(response.context["predictions"].count(), 1)

    def test_history_summary_counts_own_rows_only(self):
        self.client.login(username="owner", password="Passw0rd!xyz")
        response = self.client.get(reverse("history"))

        self.assertEqual(response.context["total"], 1)
        self.assertEqual(len(response.context["summary"]), 1)
        self.assertEqual(response.context["summary"][0]["emotion"], "joy")

    def test_empty_history_shows_empty_state(self):
        User.objects.create_user("fresh", password="Passw0rd!xyz")
        self.client.login(username="fresh", password="Passw0rd!xyz")
        response = self.client.get(reverse("history"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No predictions yet")

    def test_history_requires_login(self):
        response = self.client.get(reverse("history"))
        self.assertEqual(response.status_code, 302)


class SeedCommandTests(TestCase):
    def test_seed_creates_accounts_and_predictions(self):
        call_command("seed_db", stdout=StringIO())

        usernames = set(User.objects.values_list("username", flat=True))
        self.assertTrue({"admin", "maya", "ravi"}.issubset(usernames))

        admin = User.objects.get(username="admin")
        self.assertTrue(admin.is_superuser)

        self.assertEqual(Prediction.objects.count(), 25)
        emotions = set(Prediction.objects.values_list("predicted_emotion", flat=True))
        self.assertEqual(emotions, set(LABELS))

    def test_seed_backdates_timestamps_over_two_weeks(self):
        call_command("seed_db", stdout=StringIO())
        dates = list(Prediction.objects.values_list("created_at", flat=True))
        self.assertGreaterEqual((max(dates) - min(dates)).days, 13)

    def test_seed_is_idempotent(self):
        call_command("seed_db", stdout=StringIO())

        out = StringIO()
        call_command("seed_db", stdout=out)

        self.assertEqual(Prediction.objects.count(), 25)
        self.assertIn("Already seeded", out.getvalue())
