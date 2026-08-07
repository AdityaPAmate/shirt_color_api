from django.urls import path
from api.views import replace_fabric_view

urlpatterns = [
    path("replace-fabric/", replace_fabric_view, name="replace_fabric"),
]