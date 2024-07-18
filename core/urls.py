from django.urls import path, re_path
from . import views
from core.views import index,translate, caption, generation
from . import consumers
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.index, name='index'),
    path('caption/', views.caption, name='caption'),
    path('generation/', views.generation, name='Text'),
    path('translation/', views.translate, name='Translation'),
    path('sentiment/', views.sentiment, name='sentiment'),

] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

websocket_urlpatterns = [
    re_path(r'ws/socket-server/$', consumers.captionConsumer.as_asgi()),
    re_path(r'ws/generation-server/$', consumers.TextGenerationConsumer.as_asgi()),
    re_path(r'ws/trans-server/$', consumers.TranslationConsumer.as_asgi()),
    re_path(r'ws/sentiment-server/$', consumers.SentimentConsumer.as_asgi()),
    
]