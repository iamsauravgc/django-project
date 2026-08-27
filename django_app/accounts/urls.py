from django.urls import path
from . import views

# TODO E4-T3: wire to django.contrib.auth.views or custom views
urlpatterns = [
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
]
