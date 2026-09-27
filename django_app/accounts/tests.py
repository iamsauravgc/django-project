from django.test import Client, TestCase
from django.urls import reverse


class RegisterTests(TestCase):
    def test_register_creates_user_and_logs_in(self):
        response = self.client.post(
            reverse("register"),
            {
                "username": "newuser",
                "password1": "S3curePass!xyz",
                "password2": "S3curePass!xyz",
            },
        )
        self.assertRedirects(response, reverse("predict"))
        self.assertIn("_auth_user_id", self.client.session)

    def test_register_duplicate_username_shows_error(self):
        self.client.post(
            reverse("register"),
            {
                "username": "taken",
                "password1": "S3curePass!xyz",
                "password2": "S3curePass!xyz",
            },
        )
        self.client.logout()
        response = self.client.post(
            reverse("register"),
            {
                "username": "taken",
                "password1": "AnotherPass!xyz",
                "password2": "AnotherPass!xyz",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "errorlist")

    def test_register_password_mismatch_shows_error(self):
        response = self.client.post(
            reverse("register"),
            {
                "username": "mismatch",
                "password1": "S3curePass!xyz",
                "password2": "DifferentPass!xyz",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "errorlist")

    def test_logged_in_user_is_redirected_from_register(self):
        self.client.post(
            reverse("register"),
            {
                "username": "already",
                "password1": "S3curePass!xyz",
                "password2": "S3curePass!xyz",
            },
        )
        response = self.client.get(reverse("register"))
        self.assertRedirects(response, reverse("predict"))


class LoginTests(TestCase):
    def setUp(self):
        self.client.post(
            reverse("register"),
            {
                "username": "alice",
                "password1": "S3curePass!xyz",
                "password2": "S3curePass!xyz",
            },
        )
        self.client.logout()

    def test_valid_login_redirects_home(self):
        response = self.client.post(
            reverse("login"), {"username": "alice", "password": "S3curePass!xyz"}
        )
        self.assertRedirects(response, reverse("predict"))
        self.assertIn("_auth_user_id", self.client.session)

    def test_bad_password_renders_form_with_error(self):
        response = self.client.post(
            reverse("login"), {"username": "alice", "password": "wrongpass"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "errorlist")
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_login_honours_safe_next_parameter(self):
        response = self.client.post(
            reverse("login") + "?next=/history/",
            {"username": "alice", "password": "S3curePass!xyz"},
        )
        self.assertRedirects(response, reverse("history"))

    def test_login_ignores_unsafe_next_parameter(self):
        response = self.client.post(
            reverse("login") + "?next=https://evil.example.com/",
            {"username": "alice", "password": "S3curePass!xyz"},
        )
        self.assertRedirects(response, reverse("predict"))

    def test_logged_in_user_is_redirected_from_login(self):
        self.client.login(username="alice", password="S3curePass!xyz")
        response = self.client.get(reverse("login"))
        self.assertRedirects(response, reverse("predict"))


class LogoutTests(TestCase):
    def setUp(self):
        self.client.post(
            reverse("register"),
            {
                "username": "bob",
                "password1": "S3curePass!xyz",
                "password2": "S3curePass!xyz",
            },
        )

    def test_logout_post_clears_session_and_redirects(self):
        response = self.client.post(reverse("logout"))
        self.assertRedirects(response, reverse("login"))
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_logout_get_is_not_allowed(self):
        response = self.client.get(reverse("logout"))
        self.assertEqual(response.status_code, 405)


class ProtectedRouteTests(TestCase):
    def test_anonymous_home_redirects_to_login(self):
        response = self.client.get(reverse("predict"))
        self.assertRedirects(
            response,
            f"{reverse('login')}?next={reverse('predict')}",
        )

    def test_anonymous_history_redirects_to_login(self):
        response = self.client.get(reverse("history"))
        self.assertRedirects(
            response,
            f"{reverse('login')}?next={reverse('history')}",
        )

    def test_logged_in_user_can_open_home(self):
        self.client.post(
            reverse("register"),
            {
                "username": "carol",
                "password1": "S3curePass!xyz",
                "password2": "S3curePass!xyz",
            },
        )
        response = self.client.get(reverse("predict"))
        self.assertEqual(response.status_code, 200)
