from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_file, name='upload_file'),
    path('patients/', views.patient_list, name='patient_list'),
    path('visualize/', views.visualize_data, name='visualize_data'),
    path('get_graph_data/<str:graph_type>/', views.generate_graph, name='generate_graph'),
    path('model/', views.modelo, name='model'),
]