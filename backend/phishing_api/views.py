from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .ml_model import predict_email


@csrf_exempt
def predict(request):

    if request.method == "POST":

        data = json.loads(request.body)

        email_text = data.get("email")

        result = predict_email(email_text)

        return JsonResponse({
            "prediction": result
        })