from django.urls import path

from apps.chatbot import api, views

app_name = 'chatbot'

urlpatterns = [
    # Historia rozmów (HTML)
    path('rozmowy/', views.session_list, name='session_list'),
    path('rozmowy/ogolna/', views.session_general, name='session_general'),
    path('rozmowy/<int:pk>/', views.session_detail, name='session_detail'),
    path('rozmowy/<int:pk>/usun/', views.session_clear, name='session_clear'),
    # Widżet czatu: token JWT z sesji, potem API chronione JWT
    path('token/', api.api_token, name='api_token'),
    path('api/history/', api.api_history, name='api_history'),
    path('api/message/', api.api_send_message, name='api_send_message'),
    path('api/clear/', api.api_clear, name='api_clear'),
]
