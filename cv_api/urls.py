"""cv_api URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
import fungi_classification.views
import statisticalAnalysis.views

urlpatterns = [
    # URLs for statistical Analysis
    url(r'^statisticalAnalysis/$', statisticalAnalysis.views.simple_upload),


    # URLs for fungi classification

    url(r'^fungi_classification/predictFungiClass/examples/(?P<num>\w{0,50})$',
        fungi_classification.views.predictFungiClassUsingOverfeatViewExamples, name='examples'),
    url(r'^fungi_classification/predictFungiClass/$', fungi_classification.views.predictFungiClassUsingOverfeatView),
    url(r'^fungi_classification/predictFungiClassUsingOverfeat/$', fungi_classification.views.predictFungiClassUsingOverfeat),
    url(r'^fungi_classification/predictFungiClassUsingOverfeatWithControl/$', fungi_classification.views.predictFungiClassUsingOverfeatWithControl),
    url(r'^fungi_classification/predictFungiClassUsingGoogleNetWithControl/$', fungi_classification.views.predictFungiClassUsingGoogleNetWithControl),
    url(r'^fungi_classification/predictFungiClassUsingGoogleNet/$', fungi_classification.views.predictFungiClassUsingGoogleNet),

    url(r'^admin/', admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)