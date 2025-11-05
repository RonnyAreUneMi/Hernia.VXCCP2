from django import forms
from app.Hernia.models import Imagen
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserChangeForm
from django.contrib.auth.models import User
from .models import Profile
from django import forms
from django import forms
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.contrib.auth.forms import PasswordResetForm
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.core.mail import send_mail
from django.conf import settings
from django.templatetags.static import static

class ImagenForm(forms.ModelForm):
    class Meta:
        model = Imagen  
        fields = ['titulo', 'paciente_nombre', 'imagen']

    def __init__(self, *args, **kwargs):
        super(ImagenForm, self).__init__(*args, **kwargs)
        self.fields['titulo'].widget.attrs.update({'class': 'mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm'})
        
        self.fields['paciente_nombre'].widget.attrs.update({'class': 'mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm'})

        self.fields['imagen'].widget.attrs.update({'class': 'mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm'})


class RegistroForm(forms.ModelForm):
    email = forms.EmailField(label='Correo Electrónico')
    password1 = forms.CharField(widget=forms.PasswordInput, label='Contraseña')
    password2 = forms.CharField(widget=forms.PasswordInput, label='Confirmar Contraseña')

    class Meta:
        model = User
        fields = ['username','email', 'password1', 'password2']

    def clean_password2(self):
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')
        if password1 and password2 and password1 != password2:
            raise ValidationError("Las contraseñas no coinciden")
        return password2

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if User.objects.filter(username=username).exists():
            raise ValidationError("El nombre de usuario ya está en uso")
        return username
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise ValidationError("El correo electrónico ya está en uso")
        return email

    def save(self, commit=True):
        user = super(RegistroForm, self).save(commit=False)
        user.set_password(self.cleaned_data['password1'])
        if commit:
            user.save()
        return user

import math
from django import forms
from django.core.exceptions import ValidationError
from .models import Profile

def verificar_cedula(cedula=""):
    if len(cedula) != 10:
        raise Exception("Error: número de cédula incompleto")
    else:
        multiplicador = [2, 1, 2, 1, 2, 1, 2, 1, 2]
        ced_array = list(map(int, list(cedula)))[:9]
        ultimo_digito = int(cedula[9])
        resultado = []
        arr = zip(ced_array, multiplicador)
        
        for (i, j) in arr:
            if i * j < 10:
                resultado.append(i * j)
            else:
                resultado.append((i * j) - 9)

        if ultimo_digito == (math.ceil(float(sum(resultado)) / 10) * 10) - sum(resultado):
            return True
        else:
            raise ValidationError("Cédula inválida")
class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['profile_image','address', 'phone_number', 'cedula']

    def __init__(self, *args, **kwargs):
        super(ProfileForm, self).__init__(*args, **kwargs)
        self.fields['profile_image'].widget.attrs.update({'class': 'mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm'})
    
    def clean_cedula(self):
        cedula = self.cleaned_data.get('cedula')
        if cedula:
            verificar_cedula(cedula)
        return cedula


class UserForm(UserChangeForm):
    class Meta:
        model = User
        fields = ['username', 'email']  
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exclude(pk=self.instance.pk).exists():
            raise forms.ValidationError('El correo electrónico ya está en uso.')
        return email
        


class CustomPasswordResetForm(PasswordResetForm):
    def send_mail(self, subject_template_name, email_template_name, context, from_email, to_email, html_email_template_name=None):
        context['logo_url'] = static('images/log-hernia.png')
   
        subject = render_to_string(subject_template_name, context)
        subject = ''.join(subject.splitlines())  


        email_message = render_to_string(email_template_name, context)
        plain_message = strip_tags(email_message) 

        send_mail(
            subject,
            plain_message,
            from_email,
            [to_email],
            html_message=email_message,  
            fail_silently=False,
        )

