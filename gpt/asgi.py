"""
ASGI config for your_project_name project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/asgi/
"""

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django.urls import path
from core import consumers
# /home/shashank/Documents/GPT/gpt/core/routing.py

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gpt.settings')

# application = get_asgi_application()

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": URLRouter([
            path("ws/socket-server/", consumers.captionConsumer.as_asgi()),
            path("ws/trans-server/", consumers.TranslationConsumer.as_asgi()),
            path("ws/generation-server/", consumers.TextGenerationConsumer.as_asgi()),
            path("ws/sentiment-server/", consumers.SentimentConsumer.as_asgi()),
        ]),
})