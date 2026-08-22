# api/urls.py
from django.urls import path
from api.views import ReplaceFabricView

urlpatterns = [
    path("replace-fabric/", ReplaceFabricView.as_view(), name="replace-fabric"),
]