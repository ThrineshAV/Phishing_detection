from django.urls import path
from .views import predict_email_view

urlpatterns = [
    path("predict/", predict_email_view, name="predict_email"),
]
