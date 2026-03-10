from django.urls import path
from .views import predict_email

urlpatterns = [
    path("predict/", predict_email, name="predict_email"),
]
