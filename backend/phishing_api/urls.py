from django.urls import path
from .views import check_email

urlpatterns = [
    path("predict/", check_email, name="predict_email"),
]
