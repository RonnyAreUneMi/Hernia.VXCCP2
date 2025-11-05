# ==================== IMPORTS ====================
# Librerías estándar
import os
import hashlib
import tempfile
from io import BytesIO

# Django core
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.views import PasswordResetCompleteView
from django.core.exceptions import ObjectDoesNotExist
from django.core.files.storage import default_storage
from django.core.mail import send_mail
from django.core.paginator import Paginator
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.cache import cache_control

# ReportLab
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

# Procesamiento de imágenes
from PIL import Image, ImageDraw
import cv2
import numpy as np

# Utilidades
import pytz
import requests
from inference_sdk import InferenceHTTPClient

# Locales
from .forms import ImagenForm, RegistroForm, ProfileForm, UserForm
from .models import Imagen, Profile, Historial


# ==================== CONFIGURACIÓN ====================
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="8HZzIhc5cRGKVeheO0R7"
)


# ==================== AUTENTICACIÓN ====================
class CustomPasswordResetCompleteView(PasswordResetCompleteView):
    def get(self, request, *args, **kwargs):
        return redirect('login')


@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        try:
            user = User.objects.get(email=email)
            user = authenticate(request, username=user.username, password=password)
            if user is not None:
                login(request, user)
                return redirect('index')
            else:
                messages.error(request, 'Correo electrónico o contraseña incorrectos.')
        except User.DoesNotExist:
            messages.error(request, 'Correo electrónico o contraseña incorrectos.')
    
    return render(request, 'login.html')


def register_view(request):
    if request.method == 'POST':
        form = RegistroForm(request.POST)
        if form.is_valid():
            try:
                user = form.save()
                login(request, user)
                messages.success(request, '¡Registro exitoso! Has iniciado sesión automáticamente.')
                return redirect('index')
            except Exception as e:
                messages.error(request, f'Error al registrar el usuario: {e}')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
    else:
        form = RegistroForm()
    
    return render(request, 'register.html', {'form': form})


@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def logout_view(request):
    logout(request)
    return redirect('login')


def password_reset_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        try:
            user = User.objects.get(email=email)
            messages.success(request, 'Se ha enviado un enlace para restablecer la contraseña a tu correo electrónico.')
            return redirect('password_reset')
        except User.DoesNotExist:
            messages.error(request, 'El correo electrónico no está registrado en nuestro sistema.')
    return render(request, 'password_reset.html')


# ==================== VISTAS PRINCIPALES ====================
@login_required
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def index(request):
    if not request.session.get('welcome_message_shown', False):
        request.session['welcome_message_shown'] = True
        show_welcome_message = True
    else:
        show_welcome_message = False

    return render(request, 'home.html', {
        'user': request.user,
        'show_welcome_message': show_welcome_message,
    })


@login_required
def resultados(request):
    return render(request, 'resultados.html')


@login_required
def ver_resultado(request, id):
    historial_item = get_object_or_404(Historial, id=id, user=request.user)
    
    ecuador_tz = pytz.timezone('America/Guayaquil')
    fecha_imagen_local = historial_item.fecha_imagen.astimezone(ecuador_tz)
    
    context = {
        'grupo': historial_item.grupo,
        'porcentaje': historial_item.porcentaje,
        'processed_image_url': historial_item.imagen.url,
        'fecha_imagen': fecha_imagen_local,
        'paciente_nombre': historial_item.paciente_nombre,
        'historial_id': historial_item.id
    }
    
    return render(request, 'resultados.html', context)


# ==================== PERFIL DE USUARIO ====================
@login_required
def profile_view(request):
    if request.method == 'POST':
        user_form = UserForm(request.POST, instance=request.user)
        profile_form = ProfileForm(request.POST, request.FILES, instance=request.user.profile)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, 'Perfil actualizado exitosamente.')
            return redirect('profile')
    else:
        user_form = UserForm(instance=request.user)
        profile_form = ProfileForm(instance=request.user.profile)

    return render(request, 'perfil.html', {'user_form': user_form, 'profile_form': profile_form})


@login_required
def editar_perfil(request):
    if request.method == 'POST':
        user_form = UserForm(request.POST, instance=request.user)
        profile_form = ProfileForm(request.POST, request.FILES, instance=request.user.profile)
        
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, 'Perfil actualizado exitosamente')
            return redirect('profile')
    else:
        user_form = UserForm(instance=request.user)
        profile_form = ProfileForm(instance=request.user.profile)

    return render(request, 'editar_perfil.html', {'user_form': user_form, 'profile_form': profile_form})


@login_required
def user_profile_view(request):
    user = request.user
    profile = getattr(user, 'profile', None)

    if profile is None:
        profile = Profile.objects.create(user=user)

    return render(request, 'perfil.html', {'user': user, 'profile': profile})


# ==================== HISTORIAL MÉDICO ====================
@login_required
def historial_medico(request):
    historial = Historial.objects.filter(user=request.user).order_by('-fecha_imagen')
    paginator = Paginator(historial, 4)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request, 'historial_medico.html', {'page_obj': page_obj})


@login_required
def historial_medico_general(request):
    historial = Historial.objects.all().order_by('-fecha_imagen')
    paginator = Paginator(historial, 4)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'historial_medico_general.html', {'page_obj': page_obj})


@login_required
def eliminar_historial(request, id):
    historial_item = get_object_or_404(Historial, id=id, user=request.user)

    if request.method == "POST":
        if historial_item.imagen:
            historial_item.imagen.delete()
        
        historial_item.delete()
        messages.success(request, 'El registro ha sido eliminado correctamente.')

    return redirect('historial_med')


@login_required
def eliminar_historial_general(request, id):
    historial_item = get_object_or_404(Historial, id=id)

    if request.method == "POST":
        if historial_item.imagen:
            historial_item.imagen.delete()
        
        historial_item.delete()
        messages.success(request, 'El registro ha sido eliminado correctamente.')

    return redirect('historial_med_gene')


def guardar_resultados(request):
    if request.method == 'POST':
        historial = Historial(
            user=request.user,
            paciente_nombre=request.POST.get('paciente_nombre'),
            imagen=request.FILES.get('imagen'),
            porcentaje=request.POST.get('porcentaje'),
            grupo=request.POST.get('grupo'),
            pdf_url=request.POST.get('pdf_url')
        )
        historial.save()
        return redirect('resultados')


# ==================== PROCESAMIENTO DE IMÁGENES ====================
def subir_imagen(request):
    if request.method == 'POST':
        form = ImagenForm(request.POST, request.FILES)
        if form.is_valid():
            imagen_obj = form.save(commit=False)
            paciente_nombre = form.cleaned_data.get('paciente_nombre')
            original_name = request.FILES['imagen'].name
            hash_object = hashlib.sha256(original_name.encode())
            encrypted_name = hash_object.hexdigest() + '.' + original_name.split('.')[-1]
            imagen_obj.imagen.name = encrypted_name
            
            imagen_obj.paciente_nombre = paciente_nombre
            imagen_obj.save()

            image_url = imagen_obj.imagen.url
            response = requests.get(image_url)
            image_data = BytesIO(response.content)

            image_pil = Image.open(image_data).convert('RGB')
            img_cv2 = np.array(image_pil)
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

            result = CLIENT.infer(image_url, model_id="proy_2/1")
            predictions = result.get('predictions', [])

            for pred in predictions:
                confidence = pred['confidence'] * 100
                class_name = pred['class']

                if 'points' in pred:
                    points = np.array([[p['x'], p['y']] for p in pred['points']], dtype=np.int32)

                    overlay = img_cv2.copy()
                    cv2.fillPoly(overlay, [points], (0, 0, 255))
                    alpha = 0.4
                    cv2.addWeighted(overlay, alpha, img_cv2, 1 - alpha, 0, img_cv2)

                    color = (0, 255, 0) if class_name == 'Sin Hernia' else (255, 0, 0)
                    cv2.polylines(img_cv2, [points], isClosed=True, color=color, thickness=2)

                    x_min = min(points[:, 0])
                    y_min = min(points[:, 1])
                    text = f"{class_name} {confidence:.2f}%"
                    cv2.putText(img_cv2, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            img_pil_final = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
            buffer = BytesIO()
            img_pil_final.save(buffer, format="JPEG")
            buffer.seek(0)

            imagen_obj.imagen.save(encrypted_name, buffer)

            if result['predictions'] and result['predictions'][0]['class']:
                class_prediction = result['predictions'][0]['class']
                grupo = "Sin Hernia" if class_prediction == 'Sin Hernia' else "Hernia"
            else:
                grupo = "Predicción no encontrada."

            porcentaje = round(result['predictions'][0]['confidence'] * 100, 2) if result['predictions'] else 0
            
            ecuador_tz = pytz.timezone('America/Guayaquil')
            fecha_imagen_local = imagen_obj.fecha.astimezone(ecuador_tz)
            
            historial = Historial(
                user=request.user,
                imagen=imagen_obj.imagen,
                porcentaje=porcentaje,
                grupo=grupo,
                paciente_nombre=paciente_nombre,
                fecha_imagen=fecha_imagen_local,
            )
            historial.save()

            context = {
                'grupo': grupo,
                'porcentaje': porcentaje,
                'original_image_url': image_url,
                'processed_image_url': imagen_obj.imagen.url,
                'fecha_imagen': fecha_imagen_local,
                'paciente_nombre': paciente_nombre
            }

            return render(request, 'resultados.html', context)
    else:
        form = ImagenForm()

    return render(request, 'subir_imagen.html', {'form': form})


# ==================== GENERACIÓN DE PDFs ====================
def generar_pdf_fila(request, id):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    item = get_object_or_404(Historial, id=id)
    ecuador_tz = pytz.timezone('America/Guayaquil')
    fecha_imagen_local = item.fecha_imagen.astimezone(ecuador_tz)
    
    azul_oscuro = HexColor('#1a2332')
    azul_medio = HexColor('#2c3e50')
    gris_texto = HexColor('#2d3748')
    gris_linea = HexColor('#cbd5e0')
    verde_clinico = HexColor('#059669')
    rojo_clinico = HexColor('#dc2626')
    fondo_claro = HexColor('#f8fafc')
    
    # Encabezado
    p.setFillColor(azul_oscuro)
    p.rect(0, height - 0.9*inch, width, 0.9*inch, fill=1, stroke=0)
    
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 18)
    p.drawString(0.6*inch, height - 0.45*inch, "INFORME RADIOLÓGICO")
    
    p.setFont("Helvetica", 9)
    p.drawString(0.6*inch, height - 0.65*inch, "Departamento de Diagnóstico por Imagen")
    
    p.setFont("Helvetica", 8)
    p.drawRightString(width - 0.6*inch, height - 0.45*inch, f"No. {str(item.id).zfill(6)}")
    p.drawRightString(width - 0.6*inch, height - 0.65*inch, fecha_imagen_local.strftime('%d/%m/%Y - %H:%M'))
    
    p.setStrokeColor(gris_linea)
    p.setLineWidth(0.5)
    p.line(0.6*inch, height - 0.95*inch, width - 0.6*inch, height - 0.95*inch)
    
    # Imagen radiológica
    img_x = 0.6*inch
    img_y = height - 9.8*inch
    img_width = 4.2*inch
    img_height = 7.8*inch
    
    if item.imagen:
        p.setStrokeColor(gris_linea)
        p.setLineWidth(1)
        p.rect(img_x, img_y, img_width, img_height, stroke=1, fill=0)
        
        try:
            image_url = item.imagen.url
            response = requests.get(image_url)
            
            if response.status_code == 200 and response.content:
                image_data = BytesIO(response.content)
                img = Image.open(image_data).convert('RGB')
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    img.save(temp_file, format='JPEG')
                    temp_file_path = temp_file.name
                
                p.drawImage(temp_file_path, img_x + 0.05*inch, img_y + 0.05*inch, 
                          width=img_width - 0.1*inch, height=img_height - 0.1*inch, 
                          preserveAspectRatio=True, mask='auto')
                os.remove(temp_file_path)
        except Exception as e:
            p.setFillColor(gris_texto)
            p.setFont("Helvetica", 9)
            p.drawCentredString(img_x + img_width/2, img_y + img_height/2, "Imagen no disponible")
    
    p.setFillColor(azul_medio)
    p.setFont("Helvetica", 8)
    p.drawString(img_x, img_y - 0.25*inch, "Fig. 1 - Radiografía de tórax con marcación automatizada")
    
    # Información clínica
    right_x = 5.1*inch
    y_pos = height - 1.3*inch
    box_width = 2.9*inch
    
    # Datos del paciente
    p.setFillColor(fondo_claro)
    p.rect(right_x, y_pos - 1.35*inch, box_width, 1.35*inch, fill=1, stroke=0)
    
    p.setStrokeColor(gris_linea)
    p.setLineWidth(0.5)
    p.rect(right_x, y_pos - 1.35*inch, box_width, 1.35*inch, fill=0, stroke=1)
    
    p.setFillColor(azul_medio)
    p.setFont("Helvetica-Bold", 9)
    p.drawString(right_x + 0.15*inch, y_pos - 0.25*inch, "DATOS DEL PACIENTE")
    
    p.setStrokeColor(azul_medio)
    p.setLineWidth(1.5)
    p.line(right_x + 0.15*inch, y_pos - 0.35*inch, right_x + 1.5*inch, y_pos - 0.35*inch)
    
    p.setFillColor(gris_texto)
    p.setFont("Helvetica-Bold", 8)
    p.drawString(right_x + 0.15*inch, y_pos - 0.55*inch, "Paciente:")
    p.setFont("Helvetica", 8)
    p.drawString(right_x + 0.15*inch, y_pos - 0.7*inch, item.paciente_nombre)
    
    p.setFont("Helvetica-Bold", 8)
    p.drawString(right_x + 0.15*inch, y_pos - 0.9*inch, "Médico solicitante:")
    p.setFont("Helvetica", 8)
    p.drawString(right_x + 0.15*inch, y_pos - 1.05*inch, item.user.username)
    
    p.setFont("Helvetica", 7)
    p.setFillColor(HexColor('#64748b'))
    p.drawString(right_x + 0.15*inch, y_pos - 1.25*inch, 
                 f"Fecha: {fecha_imagen_local.strftime('%d/%m/%Y')} | Hora: {fecha_imagen_local.strftime('%H:%M')}")
    
    y_pos -= 1.65*inch
    
    # Hallazgos
    p.setFillColor(colors.white)
    p.rect(right_x, y_pos - 1*inch, box_width, 1*inch, fill=1, stroke=0)
    
    p.setStrokeColor(gris_linea)
    p.setLineWidth(0.5)
    p.rect(right_x, y_pos - 1*inch, box_width, 1*inch, fill=0, stroke=1)
    
    p.setFillColor(azul_medio)
    p.setFont("Helvetica-Bold", 9)
    p.drawString(right_x + 0.15*inch, y_pos - 0.25*inch, "HALLAZGOS")
    
    p.setStrokeColor(azul_medio)
    p.setLineWidth(1.5)
    p.line(right_x + 0.15*inch, y_pos - 0.35*inch, right_x + 1.1*inch, y_pos - 0.35*inch)
    
    diagnostico_color = verde_clinico if item.grupo == "Sin Hernia" else rojo_clinico
    
    p.setFillColor(diagnostico_color)
    p.circle(right_x + 0.25*inch, y_pos - 0.57*inch, 0.08*inch, fill=1, stroke=0)
    
    p.setFillColor(gris_texto)
    p.setFont("Helvetica-Bold", 11)
    p.drawString(right_x + 0.45*inch, y_pos - 0.62*inch, item.grupo.upper())
    
    p.setFont("Helvetica", 7)
    p.setFillColor(HexColor('#64748b'))
    p.drawString(right_x + 0.45*inch, y_pos - 0.78*inch, f"Confiabilidad del análisis: {item.porcentaje}%")
    
    y_pos -= 1.2*inch
    
    # Índice de confianza
    p.setFillColor(fondo_claro)
    p.rect(right_x, y_pos - 0.85*inch, box_width, 0.85*inch, fill=1, stroke=0)
    
    p.setStrokeColor(gris_linea)
    p.setLineWidth(0.5)
    p.rect(right_x, y_pos - 0.85*inch, box_width, 0.85*inch, fill=0, stroke=1)
    
    p.setFillColor(azul_medio)
    p.setFont("Helvetica-Bold", 9)
    p.drawString(right_x + 0.15*inch, y_pos - 0.25*inch, "ÍNDICE DE CONFIANZA")
    
    p.setStrokeColor(azul_medio)
    p.setLineWidth(1.5)
    p.line(right_x + 0.15*inch, y_pos - 0.35*inch, right_x + 1.6*inch, y_pos - 0.35*inch)
    
    bar_x = right_x + 0.15*inch
    bar_y = y_pos - 0.55*inch
    bar_width = box_width - 0.3*inch
    bar_height = 0.12*inch
    
    p.setFillColor(HexColor('#e2e8f0'))
    p.rect(bar_x, bar_y, bar_width, bar_height, fill=1, stroke=0)
    
    confidence_fill = bar_width * (float(item.porcentaje) / 100)
    p.setFillColor(azul_medio)
    p.rect(bar_x, bar_y, confidence_fill, bar_height, fill=1, stroke=0)
    
    p.setFillColor(azul_medio)
    p.setFont("Helvetica-Bold", 16)
    p.drawRightString(right_x + box_width - 0.15*inch, y_pos - 0.75*inch, f"{item.porcentaje}%")
    
    y_pos -= 1.05*inch
    
    # Interpretación radiológica
    p.setFillColor(colors.white)
    p.rect(right_x, y_pos - 2.6*inch, box_width, 2.6*inch, fill=1, stroke=0)
    
    p.setStrokeColor(gris_linea)
    p.setLineWidth(0.5)
    p.rect(right_x, y_pos - 2.6*inch, box_width, 2.6*inch, fill=0, stroke=1)
    
    p.setFillColor(azul_medio)
    p.setFont("Helvetica-Bold", 9)
    p.drawString(right_x + 0.15*inch, y_pos - 0.25*inch, "INTERPRETACIÓN RADIOLÓGICA")
    
    p.setStrokeColor(azul_medio)
    p.setLineWidth(1.5)
    p.line(right_x + 0.15*inch, y_pos - 0.35*inch, right_x + 2.1*inch, y_pos - 0.35*inch)
    
    p.setFillColor(gris_texto)
    p.setFont("Helvetica", 7.5)
    
    text_y = y_pos - 0.55*inch
    line_height = 0.14*inch
    
    if item.grupo == "Sin Hernia":
        texto = [
            "El análisis automatizado mediante inteligencia artificial",
            "no identifica signos radiológicos compatibles con hernia",
            "diafragmática en el estudio actual.",
            "",
            "La estructura diafragmática presenta morfología íntegra,",
            "sin evidencia de soluciones de continuidad ni protrusión",
            "de contenido abdominal hacia la cavidad torácica.",
            "",
            "RECOMENDACIONES:",
            "• Correlación clínica según sintomatología",
            "• Seguimiento imagenológico si persisten síntomas",
            "• Valoración médica especializada"
        ]
    else:
        texto = [
            "El análisis automatizado identifica hallazgos radiológicos",
            "compatibles con hernia diafragmática.",
            "",
            "Se observa posible alteración en la continuidad del",
            "diafragma con protrusión de estructuras que sugieren",
            "contenido abdominal hacia la cavidad torácica.",
            "",
            "RECOMENDACIONES PRIORITARIAS:",
            "• Evaluación médica especializada urgente",
            "• TC de tórax con contraste para caracterización",
            "• Interconsulta con cirugía torácica",
            "• Estudios complementarios según criterio clínico"
        ]
    
    for linea in texto:
        if linea.startswith("RECOMENDACIONES"):
            p.setFont("Helvetica-Bold", 7.5)
        elif linea.startswith("•"):
            p.setFont("Helvetica", 7)
        else:
            p.setFont("Helvetica", 7.5)
        
        p.drawString(right_x + 0.15*inch, text_y, linea)
        text_y -= line_height
    
    # Pie de página
    p.setStrokeColor(gris_linea)
    p.setLineWidth(0.5)
    p.line(0.6*inch, 1*inch, width - 0.6*inch, 1*inch)
    
    p.setFillColor(HexColor('#64748b'))
    p.setFont("Helvetica", 7)
    p.drawString(0.6*inch, 0.75*inch, "NOTA IMPORTANTE:")
    p.setFont("Helvetica", 6.5)
    p.drawString(0.6*inch, 0.6*inch, 
                 "Este informe ha sido generado mediante análisis automatizado con inteligencia artificial y debe ser validado por un médico radiólogo certificado.")
    p.drawString(0.6*inch, 0.47*inch, 
                 "Los resultados deben interpretarse en el contexto clínico del paciente. No sustituye el criterio médico profesional.")
    
    p.setFont("Helvetica", 6)
    p.setFillColor(HexColor('#94a3b8'))
    p.drawString(0.6*inch, 0.25*inch, f"Sistema de Análisis Radiológico Automatizado v2.0 | Informe ID: {str(item.id).zfill(6)} | Generado: {fecha_imagen_local.strftime('%d/%m/%Y %H:%M:%S')}")
    
    p.showPage()
    p.save()
    
    pdf = buffer.getvalue()
    buffer.close()
    
    response = HttpResponse(pdf, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="informe_rad_{str(item.id).zfill(6)}_{item.paciente_nombre.replace(" ", "_")}.pdf"'
    return response


def generar_pdf_general(request):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    if request.user.is_superuser:
        historiales = Historial.objects.all().order_by('-fecha_imagen')
    else:
        historiales = Historial.objects.filter(user=request.user).order_by('-fecha_imagen')
    
    ecuador_tz = pytz.timezone('America/Guayaquil')
    
    azul_oscuro = HexColor('#1a2332')
    gris_texto = HexColor('#2d3748')
    gris_linea = HexColor('#cbd5e0')
    
    for index, item in enumerate(historiales):
        if index > 0:
            p.showPage()
        
        fecha_local = item.fecha_imagen.astimezone(ecuador_tz)
        
        p.setFillColor(azul_oscuro)
        p.rect(0, height - 0.9*inch, width, 0.9*inch, fill=1, stroke=0)
        
        p.setFillColor(colors.white)
        p.setFont("Helvetica-Bold", 16)
        p.drawString(0.6*inch, height - 0.5*inch, "HISTORIAL RADIOLÓGICO")
        p.setFont("Helvetica", 8)
        p.drawRightString(width - 0.6*inch, height - 0.5*inch, f"Registro {index + 1} de {len(historiales)}")
        
        y_pos = height - 1.3*inch
        
        data = [
            ["Paciente:", item.paciente_nombre, "ID:", str(item.id).zfill(6)],
            ["Médico:", item.user.username, "Fecha:", fecha_local.strftime('%d/%m/%Y %H:%M')],
            ["Diagnóstico:", item.grupo, "Confianza:", f"{item.porcentaje}%"],
        ]
        
        table = Table(data, colWidths=[1*inch, 2.5*inch, 0.9*inch, 2.1*inch])
        table.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 8),
            ('FONT', (2, 0), (2, -1), 'Helvetica-Bold', 8),
            ('FONT', (1, 0), (1, -1), 'Helvetica', 8),
            ('FONT', (3, 0), (3, -1), 'Helvetica', 8),
            ('TEXTCOLOR', (0, 0), (-1, -1), gris_texto),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LINEABOVE', (0, 0), (-1, 0), 0.5, gris_linea),
            ('LINEBELOW', (0, -1), (-1, -1), 0.5, gris_linea),
        ]))
        
        table.wrapOn(p, width, height)
        table.drawOn(p, 0.6*inch, y_pos - 0.8*inch)
        
        y_pos -= 1.4*inch
        
        if item.imagen:
            img_width = 3.5*inch
            img_height = 5.5*inch
            margin_left = (width - img_width) / 2
            
            p.setStrokeColor(gris_linea)
            p.setLineWidth(1)
            p.rect(margin_left, y_pos - img_height, img_width, img_height, stroke=1, fill=0)
            
            try:
                image_url = item.imagen.url
                response = requests.get(image_url)
                
                if response.status_code == 200 and response.content:
                    image_data = BytesIO(response.content)
                    img = Image.open(image_data).convert('RGB')
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                        img.save(temp_file, format='JPEG')
                        temp_file_path = temp_file.name
                    
                    p.drawImage(temp_file_path, margin_left + 0.05*inch, y_pos - img_height + 0.05*inch, 
                              width=img_width - 0.1*inch, height=img_height - 0.1*inch, 
                              preserveAspectRatio=True, mask='auto')
                    os.remove(temp_file_path)
            except:
                pass
        
        p.setFillColor(HexColor('#94a3b8'))
        p.setFont("Helvetica", 7)
        p.drawCentredString(width/2, 0.4*inch, f"Generado: {fecha_local.strftime('%d/%m/%Y %H:%M:%S')}")
    
    p.save()
    
    pdf = buffer.getvalue()
    buffer.close()
    
    response = HttpResponse(pdf, content_type='application/pdf')
    if request.user.is_superuser:
        response['Content-Disposition'] = 'attachment; filename="historial_radiologico_completo.pdf"'
    else:
        response['Content-Disposition'] = f'attachment; filename="historial_rad_{request.user.username}.pdf"'
    
    return response