from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, 'index.html')


def generation(request):
    return render(request, 'Text.html')


def translate(request):
    return render(request, 'Translation.html')


def caption(request):
    return render(request, 'caption.html')


def sentiment(request):
    return render(request, 'sentiment_analysis.html')