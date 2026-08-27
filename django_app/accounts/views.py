"""
Epic 4 — Auth boilerplate (E4-T3)
TODO: register, login, logout via django.contrib.auth
"""
from django.shortcuts import render, redirect
# TODO: from django.contrib.auth import login, logout
# TODO: from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

def register_view(request):
    """TODO: UserCreationForm -> login -> redirect('/')"""
    # stub
    return render(request, "accounts/register.html", {})

def login_view(request):
    """TODO: AuthenticationForm -> login -> redirect('/')"""
    return render(request, "accounts/login.html", {})

def logout_view(request):
    """TODO: logout(request) -> redirect('login')"""
    pass
