from django.db import models

from questions.utils.utils import get_questions


class Context(models.Model):
    text = models.TextField(max_length=500)

    def __str__(self):
        return self.text


class Question(models.Model):
    context = models.OneToOneField(Context, on_delete=models.CASCADE)
    question = models.TextField(max_length=300)

    def __str__(self):
        return self.question

    def generate_question(self):
        self.question = get_questions(self.context.text)
