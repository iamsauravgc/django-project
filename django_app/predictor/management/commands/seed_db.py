from datetime import timedelta

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.utils import timezone

from predictor.models import Prediction
from predictor.utils import ArtifactMissing, predict_emotion

DEMO_ACCOUNTS = [
    ("admin", "Admin@12345", True),
    ("maya", "Demo@12345", False),
    ("ravi", "Demo@12345", False),
]

# (owner, text) - wording chosen so the classifier lands on all seven
# emotions; the stored label is still whatever the model really predicts.
SEED_TEXTS = [
    ("maya", "I am so happy right now, this is the best day of my entire year!"),
    ("ravi", "My hands are shaking and my heart is pounding, I have never been this scared."),
    ("maya", "The package should arrive on Thursday according to the tracking page."),
    ("ravi", "I hate when people interrupt like that, it is so rude and infuriating."),
    ("maya", "I was completely shocked when they showed up unannounced at my door."),
    ("ravi", "I cried the entire way home, I miss him so much it physically hurts."),
    ("maya", "That smell was absolutely revolting, I nearly threw up on the spot."),
    ("ravi", "We celebrated her birthday all weekend and everyone was laughing the whole time."),
    ("maya", "I am terrified of what the test results will say, I can barely sleep."),
    ("ravi", "I feel so lonely in this city, nobody even notices when I am gone."),
    ("maya", "I am so mad I could scream, nobody even bothered to reply to my emails."),
    ("ravi", "The kitchen was covered in grease and old food, it was utterly disgusting."),
    ("maya", "Wow, I never saw that coming, what a twist nobody could have predicted."),
    ("ravi", "I'm going to the store to pick up milk and bread on my way home."),
    ("maya", "Losing her was devastating, the house feels empty without her."),
    ("ravi", "Honestly I am afraid to even open that email, something feels wrong."),
    ("maya", "Honestly this is the most wonderful news I have heard in a long time."),
    ("ravi", "Suddenly the lights went out and everyone screamed, it was so unexpected."),
    ("maya", "Walking home alone in the dark makes me extremely nervous and I keep looking over my shoulder."),
    ("ravi", "Biting into that rotten meat made me gag, it tasted vile."),
    ("maya", "I broke down sobbing when I read the letter, some wounds just never heal."),
    ("ravi", "I am worried sick about the interview tomorrow, what if I completely blank out?"),
    ("maya", "I love how this turned out, it exceeded every expectation I had."),
    ("ravi", "I cannot believe they actually did that, I am honestly in shock."),
    ("maya", "Please review the attached document and send your comments by Friday."),
]


class Command(BaseCommand):
    help = "Seed demo users and a two-week prediction history (real model output)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Delete existing demo predictions before seeding.",
        )

    def handle(self, *args, **options):
        users = self.ensure_accounts()
        demo_users = [users["maya"], users["ravi"]]
        existing = Prediction.objects.filter(user__in=demo_users)

        if existing.exists() and not options["reset"]:
            self.stdout.write(
                self.style.WARNING(
                    f"Already seeded ({existing.count()} predictions). "
                    "Use --reset to wipe and reseed."
                )
            )
            self.print_accounts()
            return

        if options["reset"]:
            deleted, _ = existing.delete()
            self.stdout.write(f"Deleted {deleted} existing demo predictions.")

        try:
            self.seed_predictions(users)
        except ArtifactMissing as exc:
            self.stderr.write(self.style.ERROR(str(exc)))
            return

        self.print_summary()

    def ensure_accounts(self) -> dict:
        """Create the demo accounts if they don't exist yet."""
        users = {}
        for username, password, is_superuser in DEMO_ACCOUNTS:
            user = User.objects.filter(username=username).first()
            if user is None:
                if is_superuser:
                    user = User.objects.create_superuser(username, password=password)
                else:
                    user = User.objects.create_user(username, password=password)
                self.stdout.write(f"Created account '{username}'.")
            users[username] = user
        return users

    def seed_predictions(self, users: dict) -> None:
        """Run each text through the real model, then backdate the row."""
        now = timezone.now()
        span_days = 14
        total = len(SEED_TEXTS)

        for index, (owner, text) in enumerate(SEED_TEXTS):
            result = predict_emotion(text)
            prediction = Prediction.objects.create(
                user=users[owner],
                text=text,
                predicted_emotion=result["predicted"],
                confidence=result["confidence"],
                all_scores=result["scores"],
            )

            # auto_now_add ignores values passed to create(), so backdate
            # with a separate UPDATE and keep the rows in chronological order.
            age = timedelta(days=(index / total) * span_days, hours=index % 5)
            Prediction.objects.filter(pk=prediction.pk).update(created_at=now - age)

        self.stdout.write(
            self.style.SUCCESS(f"Seeded {total} predictions across 2 users.")
        )

    def print_accounts(self) -> None:
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Sign in with:"))
        for username, password, is_superuser in DEMO_ACCOUNTS:
            role = "admin superuser" if is_superuser else "demo user"
            self.stdout.write(f"  {username:<6} {password:<12} ({role})")

    def print_summary(self) -> None:
        self.print_accounts()

        rows = Prediction.objects.filter(
            user__username__in=["maya", "ravi"]
        ).values_list("predicted_emotion", flat=True)

        counts: dict[str, int] = {}
        for emotion in rows:
            counts[emotion] = counts.get(emotion, 0) + 1

        breakdown = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        self.stdout.write(f"\nEmotion breakdown: {breakdown}")
        self.stdout.write(f"Total predictions: {sum(counts.values())}")
