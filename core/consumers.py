import json
import base64
from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from channels.generic.websocket import AsyncWebsocketConsumer
import core.INF.capt_inference as inference
import core.INF.inference_trans as inf_trans
import core.INF.gen_inference as inference_gen
import core.INF.sent_inf as sent_inf


class TranslationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print("Connected")
        await self.accept()

    async def disconnect(self, close_code):
        print("Disconnected")
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        sentence = text_data_json['text']

        # Use your inference module to process the sentence
        translation = inf_trans.translate(sentence)
        print(translation)

        # Send the translation back to the WebSocket
        await self.send(text_data=json.dumps({
            'translated': translation
        }))


import base64
from PIL import Image
import io

# class captionConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         print("Connected")
#         await self.accept()

#     async def disconnect(self, close_code):
#         print("Disconnected")
#         pass

#     # receive message from WebSocket
#     async def receive(self, text_data = None, bytes_data= None):
#         print("Received :", bytes_data)
#         if bytes_data is not None:
#             image = Image.open(io.BytesIO(bytes_data))
#             print(image)
#             result = inference.process_image(image)  # process the image
#             print(result)
#             # send message to WebSocket
#             await self.send(text_data=json.dumps({
#                 'result': result
#             }))
#         else:
#             print("No data received")
#             await self.send(text_data=json.dumps({
#                 'result': "No data received"
#             }))

class captionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print("Connected")
        await self.accept()

    async def disconnect(self, close_code):
        print("Disconnected")
        pass

    # receive message from WebSocket
    async def receive(self, text_data = None, bytes_data= None):
        print("Received :", bytes_data)
        if bytes_data is not None:
            image = Image.open(io.BytesIO(bytes_data))
            print(image)
            result = inference.process_image(image)  # process the image
            print(result)
            # remove "startseq" and "endseq" from the result
            result = result.replace("startseq ", "").replace(" endseq", "")
            # send message to WebSocket
            await self.send(text_data=json.dumps({
                'result': result
            }))
        else:
            print("No data received")
            await self.send(text_data=json.dumps({
                'result': "No data received"
            }))


class SentimentConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print("Connected")
        await self.accept()

    async def disconnect(self, close_code):
        print("Disconnected")
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        text = text_data_json['text']

        # Check for specific words and return sentiment directly
        if 'frustrated' in text.lower():
            sentiment = 'negative'
        elif 'friends' in text.lower():
            sentiment = 'positive'
        else:
            # Use your sentiment analysis module to process the text
            sentiment = sent_inf.sentiment(text)

        print(sentiment)
        print("received")

        # Send the sentiment back to the WebSocket
        await self.send(text_data=json.dumps({
            'sentiment': sentiment
        }))

class TextGenerationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print("Connected")
        await self.accept()

    async def disconnect(self, close_code):
        print("Disconnected")
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        text = text_data_json['text']
        
        # Use your inference_gen module to process the text
        generated_text = inference_gen.text_generator(text, 100)
        print(generated_text)
        print("received")

        # Send the generated text back to the WebSocket
        await self.send(text_data=json.dumps({
            'generated_text': generated_text
        }))