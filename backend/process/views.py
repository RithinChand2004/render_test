from django.shortcuts import render
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
import azure.cognitiveservices.speech as speechsdk
from .models import Image
from .serializers import ImageSerializer
from django.http import JsonResponse, HttpResponseBadRequest
from azure.storage.blob import BlobServiceClient
import json
import base64
import os
import pyttsx3
import threading
import azure.ai.vision as sdk
import cv2
import numpy as np
import requests
from io import BytesIO
import mediapipe as mp
import azure.ai.vision as sdk
# Create your views here.


def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(text)
    engine.runAndWait()

def speakOut(text):
    speech_thread = threading.Thread(target=text_to_speech, args=(text,))
    speech_thread.start()

def speak(request):
    text = (request.GET.get('text'))
    speakOut(text)
    return JsonResponse({'text': text})

@csrf_exempt
def uploadFrame(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data_url = data['image']

            # Connect to your Azure Blob Storage account
            connection_string = "DefaultEndpointsProtocol=https;AccountName=bumble;AccountKey=c/mSmXiLPb7OK3KKC8WdnUPP81pCjg3ApavOfPHNvk+iPtVFfWd660mupLiJ3QIt/1sege3FYZuv+AStI/KEmA==;EndpointSuffix=core.windows.net"
            blob_service_client = BlobServiceClient.from_connection_string(
                connection_string)

            container_name = "data"
            container_client = blob_service_client.get_container_client(
                container_name)

            # Generate a unique blob name (e.g., based on timestamp or UUID)
            blob_name = "frame123.jpg"  # Replace with your naming logic

            # Decode the base64 image data and upload it to Azure Blob Storage
            image_bytes = base64.b64decode(image_data_url.split(',')[1])
            container_client.upload_blob(
                name=blob_name, data=image_bytes, overwrite=True)

            return JsonResponse({'message': 'Frame uploaded successfully'})
        except Exception as e:
            return HttpResponseBadRequest('Error uploading frame: ' + str(e))
    else:
        return HttpResponseBadRequest('Invalid request method')

def recognize_from_microphone():
    text_to_speech("How can I help you today?")
    output = ''
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(
        "a7bb5bb2551b4837a6183c173ad34e54", 'eastus')
    speech_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config)

    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
        output = speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(
            speech_recognition_result.no_match_details))
        output = "No speech could be recognized: {}".format(
            speech_recognition_result.no_match_details)
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(
            cancellation_details.reason))
        output = "Speech Recognition canceled: {}".format(
            cancellation_details.reason)
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(
                cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
    return output


def home(request):
    context = {'textOutput': 0}
    return render(request, 'home.html')

def listenAndRecognize(request):
    text = recognize_from_microphone()
    return JsonResponse({'text': text})


@csrf_exempt
def uploadImg(request):
    # csrf tockens

    if request.method == 'POST':
        print(request.POST)
        print(request.FILES)
        title = request.POST['title']
        image = request.FILES['image']
        print(title)
        print(image)
        Image.objects.create(title=title, image=image)
    return render(request, 'home.html')


class ImageUploadView(APIView):
    parser_classes = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
        print(request.data)
        file_serializer = ImageSerializer(data=request.data)
        print(file_serializer)
        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

def findObjects(request):
    service_options = sdk.VisionServiceOptions("https://dristi.cognitiveservices.azure.com/",
                                               "ec8f8825967f482699df9b8080d3d826")

    vision_source = sdk.VisionSource(
        url="https://bumble.blob.core.windows.net/data/frame123.jpg")

    analysis_options = sdk.ImageAnalysisOptions()

    analysis_options.features = (
        # sdk.ImageAnalysisFeature.CAPTION |
        # sdk.ImageAnalysisFeature.DENSE_CAPTIONS |
        sdk.ImageAnalysisFeature.OBJECTS 
        # sdk.ImageAnalysisFeature.PEOPLE |
        # sdk.ImageAnalysisFeature.TEXT |
        # sdk.ImageAnalysisFeature.TAGS
    )

    analysis_options.language = "en"

    analysis_options.gender_neutral_caption = True

    image_analyzer = sdk.ImageAnalyzer(
        service_options, vision_source, analysis_options)

    result = image_analyzer.analyze()

    if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:

        print(" Image height: {}".format(result.image_height))
        print(" Image width: {}".format(result.image_width))
        print(" Model version: {}".format(result.model_version))

        obj = ''
        if result.objects is not None:
            print(" Objects:")
            for object in result.objects:
                print("   '{}', {}, Confidence: {:.4f}".format(
                    object.name, object.bounding_box, object.confidence))
                if object.confidence > 0.5:
                    obj += object.name + ' '
        else:
            print("No objects detected.")
        
        speakOut(obj)
        return JsonResponse({'objects': obj})

        

    else:

        error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
        print(" Analysis failed.")
        # print("   Error reason: {}".format(error_details.reason))
        # print("   Error code: {}".format(error_details.error_code))
        # print("   Error message: {}".format(error_details.message))

def busroute(request):
    text = (request.GET.get('text'))
    print(text)
    text = text.lower()
    text = text.split(' ')
    city = text[-1]
    print(city)
    service_options = sdk.VisionServiceOptions("https://dristi.cognitiveservices.azure.com/",
                                               "ec8f8825967f482699df9b8080d3d826")

    vision_source = sdk.VisionSource(
        url="https://bumble.blob.core.windows.net/data/frame123.jpg")

    analysis_options = sdk.ImageAnalysisOptions()

    analysis_options.features = (
        # sdk.ImageAnalysisFeature.CAPTION |
        # sdk.ImageAnalysisFeature.DENSE_CAPTIONS |
        # sdk.ImageAnalysisFeature.OBJECTS |
        # sdk.ImageAnalysisFeature.PEOPLE |
        sdk.ImageAnalysisFeature.TEXT |
        sdk.ImageAnalysisFeature.TAGS
    )

    analysis_options.language = "en"

    analysis_options.gender_neutral_caption = True

    image_analyzer = sdk.ImageAnalyzer(
        service_options, vision_source, analysis_options)

    result = image_analyzer.analyze()
    tagt = ''

    if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:

        print(" Image height: {}".format(result.image_height))
        print(" Image width: {}".format(result.image_width))
        print(" Model version: {}".format(result.model_version))

        
        if result.tags is not None:
            print(" Tags:")
            for tag in result.tags:
                print("   '{}', Confidence {:.4f}".format(
                    tag.name, tag.confidence))
                tagt += tag.name + ' '
        
        if 'bus' in tagt:
            t = ''
            if result.text is not None:
                print(" Text:")
                for line in result.text.lines:
                    points_string = "{" + ", ".join([str(int(point))
                                                    for point in line.bounding_polygon]) + "}"
                    print("   Line: '{}', Bounding polygon {}".format(
                        line.content, points_string))
                    for word in line.words:
                        points_string = "{" + ", ".join([str(int(point))
                                                        for point in word.bounding_polygon]) + "}"
                        print("     Word: '{}', Bounding polygon {}, Confidence {:.4f}"
                            .format(word.content, points_string, word.confidence))
                        t += word.content + ' '
            t = t.lower()
            if city[:-1] in t:
                speakOut('This bus goes to ' + city)
                return JsonResponse({'bus': 'This bus goes to ' + city})
            else:
                speakOut('This bus does not go to ' + city)
                return JsonResponse({'bus': 'This bus does not go to ' + city})
        else:
            speakOut('No bus found')
            return JsonResponse({'bus': 'No bus found'})

        # result_details = sdk.ImageAnalysisResultDetails.from_result(result)
        # print(" Result details:")
        # print("   Image ID: {}".format(result_details.image_id))
        # print("   Result ID: {}".format(result_details.result_id))
        # print("   Connection URL: {}".format(result_details.connection_url))
        # print("   JSON result: {}".format(result_details.json_result))

    else:

        error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
        print(" Analysis failed.")
        # print("   Error reason: {}".format(error_details.reason))
        # print("   Error code: {}".format(error_details.error_code))
        # print("   Error message: {}".format(error_details.message))

def danger(request):
    service_options = sdk.VisionServiceOptions("https://dristi.cognitiveservices.azure.com/",
                                               "ec8f8825967f482699df9b8080d3d826")

    vision_source = sdk.VisionSource(
        url="https://bumble.blob.core.windows.net/data/frame123.jpg")

    analysis_options = sdk.ImageAnalysisOptions()

    analysis_options.features = (
        sdk.ImageAnalysisFeature.CAPTION |
        # sdk.ImageAnalysisFeature.DENSE_CAPTIONS |
        # sdk.ImageAnalysisFeature.OBJECTS |
        # sdk.ImageAnalysisFeature.PEOPLE |
        # sdk.ImageAnalysisFeature.TEXT |
        sdk.ImageAnalysisFeature.TAGS
    )

    analysis_options.language = "en"

    analysis_options.gender_neutral_caption = True

    image_analyzer = sdk.ImageAnalyzer(
        service_options, vision_source, analysis_options)

    result = image_analyzer.analyze()
    t = ''
    if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:

        print(" Image height: {}".format(result.image_height))
        print(" Image width: {}".format(result.image_width))
        print(" Model version: {}".format(result.model_version))

        if result.caption is not None:
            print(" Caption:")
            print("   '{}', Confidence {:.4f}".format(
                result.caption.content, result.caption.confidence))
        t = ''
        if result.tags is not None:
            print(" Tags:")
            for tag in result.tags:
                print("   '{}', Confidence {:.4f}".format(
                    tag.name, tag.confidence))
                print(tag.name,'weapon',tag.name == 'weapon')
                t += tag.name + ' '
                if 'weapon' == tag.name:
                    speakOut('danger ' + result.caption.content)
                    return JsonResponse({'danger': ('danger ' + result.caption.content)})
            
        
        

        # result_details = sdk.ImageAnalysisResultDetails.from_result(result)
        # print(" Result details:")
        # print("   Image ID: {}".format(result_details.image_id))
        # print("   Result ID: {}".format(result_details.result_id))
        # print("   Connection URL: {}".format(result_details.connection_url))
        # print("   JSON result: {}".format(result_details.json_result))

    else:

        error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
        print(" Analysis failed.")
        # print("   Error reason: {}".format(error_details.reason))
        # print("   Error code: {}".format(error_details.error_code))
        # print("   Error message: {}".format(error_details.message))
    return JsonResponse({'danger': f"Objects are \n {t}. \n No weapons are detected"})

def surroundings(request):
    service_options = sdk.VisionServiceOptions("https://dristi.cognitiveservices.azure.com/",
                                               "ec8f8825967f482699df9b8080d3d826")

    vision_source = sdk.VisionSource(
        url="https://bumble.blob.core.windows.net/data/frame123.jpg")

    analysis_options = sdk.ImageAnalysisOptions()

    analysis_options.features = (
        sdk.ImageAnalysisFeature.CAPTION |
        sdk.ImageAnalysisFeature.DENSE_CAPTIONS 
        # sdk.ImageAnalysisFeature.OBJECTS |
        # sdk.ImageAnalysisFeature.PEOPLE |
        # sdk.ImageAnalysisFeature.TEXT |
        # sdk.ImageAnalysisFeature.TAGS
    )

    analysis_options.language = "en"

    analysis_options.gender_neutral_caption = True

    image_analyzer = sdk.ImageAnalyzer(
        service_options, vision_source, analysis_options)

    result = image_analyzer.analyze()
    cap = ''
    if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:

        print(" Image height: {}".format(result.image_height))
        print(" Image width: {}".format(result.image_width))
        print(" Model version: {}".format(result.model_version))

        if result.caption is not None:
            print(" Caption:")
            print("   '{}', Confidence {:.4f}".format(
                result.caption.content, result.caption.confidence))
            cap = result.caption.content

        if result.dense_captions is not None:
            print(" Dense Captions:")
            for caption in result.dense_captions:
                print("   '{}', {}, Confidence: {:.4f}".format(
                    caption.content, caption.bounding_box, caption.confidence))
        
        speakOut(cap)
        return JsonResponse({'caption': cap})

        
        # result_details = sdk.ImageAnalysisResultDetails.from_result(result)
        # print(" Result details:")
        # print("   Image ID: {}".format(result_details.image_id))
        # print("   Result ID: {}".format(result_details.result_id))
        # print("   Connection URL: {}".format(result_details.connection_url))
        # print("   JSON result: {}".format(result_details.json_result))

    else:

        error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
        print(" Analysis failed.")
        # print("   Error reason: {}".format(error_details.reason))
        # print("   Error code: {}".format(error_details.error_code))
        # print("   Error message: {}".format(error_details.message))


def isInsideBox(position, x, y, w, h):
        if position[0] >= x and position[0] <= x + w:
            if position[1] >= y and position[1] <= y + h:
                return True
        return False


def retriveCaption():
    service_options = sdk.VisionServiceOptions("https://dristi.cognitiveservices.azure.com/",
                                               "ec8f8825967f482699df9b8080d3d826")

    vision_source = sdk.VisionSource(
        url="https://bumble.blob.core.windows.net/data/frame123.jpg")

    analysis_options = sdk.ImageAnalysisOptions()

    analysis_options.features = (
        sdk.ImageAnalysisFeature.CAPTION |
        sdk.ImageAnalysisFeature.DENSE_CAPTIONS
        # sdk.ImageAnalysisFeature.OBJECTS |
        # sdk.ImageAnalysisFeature.PEOPLE |
        # sdk.ImageAnalysisFeature.TEXT |
        # sdk.ImageAnalysisFeature.TAGS
    )

    analysis_options.language = "en"

    analysis_options.gender_neutral_caption = True

    image_analyzer = sdk.ImageAnalyzer(
        service_options, vision_source, analysis_options)

    result = image_analyzer.analyze()

    if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:
        return result


def fifs(request):
    image_url = "https://bumble.blob.core.windows.net/data/frame123.jpg"
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        if response.status_code != 200:
            return None

        # Read the image from the response
        image_data = BytesIO(response.content)
        img = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), -1)
        x, y, c = img.shape
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        # Perform hand and finger detection here using OpenCV

        # Example: Convert the image to grayscale
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # Example: Apply hand and finger detection algorithms
        # You would need to implement or use existing algorithms for hand and finger detection
        # This may involve techniques like contour detection, skin color detection, and finger tracking

        # use mediapipe to detect hand
        mpHands = mp.solutions.hands  # mpHands is the object which holds the data
        hands = mpHands.Hands(
            max_num_hands=1, min_detection_confidence=0.7)
        mpDraw = mp.solutions.drawing_utils  # draws the hand moveme
        textsToBeRead = []

        results = hands.process(frame)
        res = retriveCaption()
        if results.multi_hand_landmarks:
            print("Hand detected")
            landmarks = []
            for handslms in results.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS,
                                      mpDraw.DrawingSpec(
                                        color=(0, 0, 255), thickness=2, circle_radius=2),
                                      mpDraw.DrawingSpec(
                                        color=(0, 255, 0), thickness=2, circle_radius=2),
                                      )
            position = (landmarks[8])
            print(position)
            bounds = []
            if res.dense_captions is not None:
                print(" Dense Captions:")
                for caption in res.dense_captions:
                    # print("   '{}', {}, Confidence: {:.4f}".format(
                    #     caption.content, caption.bounding_box, caption.confidence))
                    # print(caption.bounding_box.x)
                    # print(caption.bounding_box.y)
                    # print(caption.bounding_box.w)
                    # print(caption.bounding_box.h)
                    # print()
                    # textsToBeRead.append(caption.content)
                    if isInsideBox(position, caption.bounding_box.x, caption.bounding_box.y, caption.bounding_box.w, caption.bounding_box.h):
                        print("Inside box")
                        print(caption.content)
                        textsToBeRead.append(caption.content)
                        bounds.append(caption.bounding_box)
                print(textsToBeRead)
                temp = ' '.join(textsToBeRead)
                speakOut("Hand detected "+temp)
                return JsonResponse({'caption': temp})
            else:
                print("No dense captions detected")
                if res.caption is not None:
                    print(" Caption:")
                    print(res.caption.content)
                    speakOut(res.caption.content)
                    return JsonResponse({'caption': res.caption.content})

        else:
            print("No hand detected")
            if res.caption is not None:
                print(" Caption:")
                print(res.caption.content)
                speakOut(res.caption.content)

                return JsonResponse({'caption': res.caption.content})

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Example usage:
# Replace with the image URL you want to process


