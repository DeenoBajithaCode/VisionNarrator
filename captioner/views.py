from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .models import generate_caption
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from django.core.files.storage import FileSystemStorage
@csrf_exempt
@require_POST
def generate_caption_view(request):
    try:
        image = request.FILES['image']
        image_path = os.path.join('media', image.name)
        with default_storage.open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)

        # Generate caption using CLIP and GPT-2
        caption = generate_caption(image_path)

        # Clean up the uploaded image
        os.remove(image_path)

        return JsonResponse({'caption': caption})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

def handle_uploaded_file(f):
    fs = FileSystemStorage()
    filename = fs.save(f.name, f)
    file_url = fs.url(filename)
    return file_url
