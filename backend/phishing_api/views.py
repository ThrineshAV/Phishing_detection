from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .ml_model import predict_email


@csrf_exempt
def predict(request):
    if request.method == "OPTIONS":
        return JsonResponse({}, status=200)

    if request.method == "GET":
        email_text = request.GET.get("email", "").strip()

        if not email_text:
            return JsonResponse(
                {
                    "message": "Send a POST request with JSON {'email': '...'} or a GET request with ?email=...",
                },
                status=200,
            )

        try:
            result = predict_email(email_text)
        except Exception as exc:
            return JsonResponse({"error": str(exc)}, status=500)
        return JsonResponse({"prediction": result})

    if request.method == "POST":
        try:
            data = json.loads(request.body or "{}")
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON payload."}, status=400)

        email_text = data.get("email", "").strip()

        if not email_text:
            return JsonResponse({"error": "The 'email' field is required."}, status=400)

        try:
            result = predict_email(email_text)
        except Exception as exc:
            return JsonResponse({"error": str(exc)}, status=500)

        return JsonResponse({
            "prediction": result
        })

    return JsonResponse({"error": "Only GET and POST requests are allowed."}, status=405)


predict_email_view = predict
