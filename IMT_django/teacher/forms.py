from django import forms

class consent_check(forms.Form):
    check = forms.BooleanField(required = True)