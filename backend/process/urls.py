from django.urls import path
from . import views
from .views import ImageUploadView

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', ImageUploadView.as_view(), name='image-upload'),
    path('uploadImg/', views.uploadImg, name='uploadImg'),
    path('listenAndRecognize/', views.listenAndRecognize, name='listenAndRecognize'),
    path('uploadFrame/', views.uploadFrame, name='uploadFrame'),
    path('findObjects/', views.findObjects, name='findObjects'),
    path('busroute/', views.busroute, name='busroute'),
    path('danger/', views.danger, name='danger'),
    path('surroundings/', views.surroundings, name='surroundings'),
    path('speak', views.speak, name='speak'),
    path('fifs/', views.fifs, name='fifs'),

]
