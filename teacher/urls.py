from django.urls import path
from . import views

# admin.autodiscover()

# app_name = 'teacher'
urlpatterns = [
	path('', views.index, name='index'),
	# path('save_recording/', views.save_recording, name='save_recording'),
	path('feedback/', views.feedback, name='feedback'),
	path('processTeachingAnswer/', views.processTeachingAnswer,
		 name='processTeachingAnswer'),
    path('processTestingAnswer/', views.processTestingAnswer,
		 name='processTestingAnswer'),
]