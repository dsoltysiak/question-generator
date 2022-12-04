from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .models import Context, Question
from .forms import AddContext


def context(response):
    if response.method == "POST":
        form = AddContext(response.POST)
        if form.is_valid():
            t = form.cleaned_data["text"]
            input = Context(text=t)
            input.save()
        return HttpResponseRedirect("/%i" % input.id)
    else:
        form = AddContext()
    return render(response, "questions/post_text.html", {"form": form})


def question(response, id):
    print(response.method)
    # if response.method == "POST":
    #     form = AddContext(response.POST)
    #     if form.is_valid():
    #         t = form.cleaned_data["text"]
    #         input = Context(text=t)
    #         input.save()
    #     return HttpResponseRedirect("/%i" % input.id)
    # else:
    #     form = AddContext()
    context = Context.objects.get(id=id)
    # HERE CALL FOR THE QUESTION GENERATION, OR MAKE IT AS CLASS METHOD IN QUESTION
    # question = Question(context=context, question=f"Question for input number {id}")
    question = Question(context=context)
    question.generate_question()
    return render(response, "questions/show_question.html", {"question": question})
