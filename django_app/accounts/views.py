"""
Epic 4 — Auth views (E4-T3)
"""
from django.contrib.auth import login, logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.shortcuts import redirect, render
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST


def _redirect_target(request, fallback="predict"):
    """Honour ?next= only when it is a safe local URL; else fall back."""
    target = request.POST.get("next") or request.GET.get("next")
    if target and url_has_allowed_host_and_scheme(target, allowed_hosts={request.get_host()}):
        return target
    return fallback


@never_cache
def register_view(request):
    if request.user.is_authenticated:
        return redirect(_redirect_target(request))

    next_url = _redirect_target(request, fallback=None)
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect(next_url or "predict")
    else:
        form = UserCreationForm()

    return render(request, "accounts/register.html", {"form": form, "next": next_url})


@never_cache
def login_view(request):
    if request.user.is_authenticated:
        return redirect(_redirect_target(request))

    next_url = _redirect_target(request, fallback=None)
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            return redirect(next_url or "predict")
    else:
        form = AuthenticationForm()

    return render(request, "accounts/login.html", {"form": form, "next": next_url})


@require_POST
@csrf_protect
@never_cache
def logout_view(request):
    logout(request)
    return redirect("login")
