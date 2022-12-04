from django import forms


class AddContext(forms.Form):
    text = forms.CharField(
        max_length=200,
        label=False,
        widget=forms.Textarea(
            attrs={"placeholder": "Enter your text here", "class": "box"}
        ),
    )
