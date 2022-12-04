from django.urls import path

from . import views

urlpatterns = [
    # path("<int:id>", views.index, name="index"),
    path("", views.context, name="context"),
    path("<int:id>", views.question, name="question"),
]
