from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from statisticalComparison import statisticalComparison

def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        report = statisticalComparison.statisticalComparison("/home/joheras/pythonprojects/api/cv_api"+uploaded_file_url)
        return render(request, 'statisticalAnalysis/simple_upload.html', {
            'uploaded_file_url': uploaded_file_url, 'report' : report
        })
    return render(request, 'statisticalAnalysis/simple_upload.html')