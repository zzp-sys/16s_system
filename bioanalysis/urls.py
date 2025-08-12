from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('get_log/', views.get_log, name='get_log'),
    path('clear_log/', views.clear_log, name='clear_log'),
    path('stop_analysis/', views.stop_analysis, name='stop_analysis'),
    path('get_available_samples/', views.get_available_samples, name='get_available_samples'),
    path('get_user_stats/', views.get_user_stats_api, name='get_user_stats'),
    path('integrated/', views.integrated_analysis, name='integrated_analysis'),
    path('sample_results/', views.sample_results, name='sample_results'),
    path('predict_diseases/', views.predict_diseases, name='predict_diseases'),
]
