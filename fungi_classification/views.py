import time
import datetime
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import cv2
from overfeat.predict import categoryOfFungiImage as categoryOfFungiImageUsingOverfeat
from overfeatwithcontrol.predict import categoryOfFungiImage as categoryOfFungiImageUsingOverfeatWithControl
from googlenetwithcontrol.predict import categoryOfFungiImage as categoryOfFungiImageUsingGoogleNetWithControl
from googlenet.predict import categoryOfFungiImage as categoryOfFungiImageUsingGoogleNet
from overfeat.model.overfeat import OverfeatExtractor
from googlenet.model.googlenet import GoogleNetExtractor
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import json
import os
import zipfile

# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))

oe = OverfeatExtractor()
#ge = GoogleNetExtractor()


@csrf_exempt
def predictFungiClassUsingOverfeat(request):
    """CURL example: curl -X POST -F image=@ad71aem_control.jpg 'http://localhost:8000/fungi_classification/predictFungiClassUsingOverfeat/'; echo "" """

    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("url", None)

            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            # load the image and convert
            image = _grab_image(url=url)

        # predict the class of the fungi image
        prediction = categoryOfFungiImageUsingOverfeat(image,oe)

        # Saving the file for further usage
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        extension = myfile.name[myfile.name.rfind("."):]
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        filename = fs.save("fungi-classification/"+st + extension, myfile)
        uploaded_file_url = fs.url(filename)

        # update the data dictionary with the categories
        data.update({"prediction": prediction})
        data.update({"success": True})
        data.update({"uploaded_file_url":uploaded_file_url})

    # return a JSON response
    return JsonResponse(data)

@csrf_exempt
def predictFungiClassUsingOverfeatView(request):
    if request.method == 'POST':
        """and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        report = statisticalComparison.statisticalComparison(BASE_DIR+uploaded_file_url)
        fs.delete(filename)"""
        print(request.POST)
        predictionJSON = predictFungiClassUsingOverfeat(request)

        return render(request, 'FungiClassification/simple_upload.html', json.loads(predictionJSON.content))
    return render(request, 'FungiClassification/simple_upload.html')



@csrf_exempt
def predictFungiClassUsingOverfeatViewZIP(request):
    if request.method == 'POST':
        zip_file = request.FILES['zip']
        fs = FileSystemStorage()
        extension = zip_file.name[zip_file.name.rfind("."):]
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        fs.save("zip-files/" + st + extension, zip_file)
        zip_ref = zipfile.ZipFile("media/zip-files/"+ st + extension, 'r')
        zip_ref.extractall("media/zip-files/"+ st)
        zip_ref.close()
        fs.delete("zip-files/" + st + extension)
        (_,files)=fs.listdir("zip-files/"+ st)
        print(files)
        predictions = [("../../../media/zip-files/"+st+"/"+file,categoryOfFungiImageUsingOverfeat(_grab_image(path="media/zip-files/"+ st+"/"+file), oe)) for file in files]
        data = {"success": True}
        data.update({"predictions": predictions})
        predictionJSON = JsonResponse(data)

        return render(request, 'FungiClassification/zip_upload.html', json.loads(predictionJSON.content))
    return render(request, 'FungiClassification/zip_upload.html')


@csrf_exempt
def predictFungiClassUsingOverfeatViewExamples(request,num=1):
    image = _grab_image(path="media/fungi-classification/" + num + ".jpg")
    prediction = categoryOfFungiImageUsingOverfeat(image, oe)

    # update the data dictionary with the categories
    data = {"success": True}
    data.update({"prediction": prediction})
    data.update({"uploaded_file_url": "../../../media/fungi-classification/" + num + ".jpg"})
    predictionJSON = JsonResponse(data)

    return render(request, 'FungiClassification/simple_upload.html', json.loads(predictionJSON.content))



@csrf_exempt
def predictFungiClassUsingOverfeatWithControl(request):
    """CURL example: curl -X POST -F image=@ad71aem_control.jpg -F imageControl=@ad71aem_control.jpg 'http://localhost:8000/fungi_classification/predictFungiClassUsingOverfeatWithControl/'; echo "" """

    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None and request.FILES.get("imageControl", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])
            imageControl = _grab_image(stream=request.FILES["imageControl"])
        else:
            data.update({"Error": "You must upload two images"})
            return JsonResponse(data)

        # predict the class of the fungi image
        prediction = categoryOfFungiImageUsingOverfeatWithControl(image,imageControl,oe)

        # update the data dictionary with the faces detected
        data.update({"prediction": prediction})
        data.update({"success": True})

    # return a JSON response
    return JsonResponse(data)



@csrf_exempt
def predictFungiClassUsingGoogleNet(request):
    """CURL example: curl -X POST -F image=@ad71aem_control.jpg 'http://localhost:8000/fungi_classification/predictFungiClassUsingGoogleNet/'; echo "" """
    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("url", None)

            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            # load the image and convert
            image = _grab_image(url=url)

        # predict the class of the fungi image
        prediction = categoryOfFungiImageUsingGoogleNet(image,ge)

        # update the data dictionary with the faces detected
        data.update({"prediction": prediction})
        data.update({"success": True})

    # return a JSON response
    return JsonResponse(data)


@csrf_exempt
def predictFungiClassUsingGoogleNetWithControl(request):
    """CURL example: curl -X POST -F image=@ad71aem_control.jpg -F imageControl=@ad71aem_control.jpg 'http://localhost:8000/fungi_classification/predictFungiClassUsingGoogleNetWithControl/'; echo "" """

    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None and request.FILES.get("imageControl", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])
            imageControl = _grab_image(stream=request.FILES["imageControl"])
        else:
            data.update({"Error": "You must upload two images"})
            return JsonResponse(data)

        # predict the class of the fungi image
        prediction = categoryOfFungiImageUsingGoogleNetWithControl(image,imageControl,ge)

        # update the data dictionary with the faces detected
        data.update({"prediction": prediction})
        data.update({"success": True})

    # return a JSON response
    return JsonResponse(data)


def _grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    print path
    if path is not None:
        image = cv2.imread(path)

    # otherwise, the image does not reside on disk
    else:
        # if the URL is not None, then download the image
        if url is not None:
            resp = urllib.urlopen(url)
            data = resp.read()

        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()

        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image