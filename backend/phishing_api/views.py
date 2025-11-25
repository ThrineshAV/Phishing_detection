from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .ml_model import predict_email

@csrf_exempt
def check_email(request):
    if request.method == "POST":
        data = json.loads(request.body)
        email_text = data.get("email_text", "")
        result = predict_email(email_text)
        return JsonResponse({"result": result})
    return JsonResponse({"error": "Only POST method allowed"}, status=405)
