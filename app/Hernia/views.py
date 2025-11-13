# ==================== VISTAS ====================
"""
Módulo de vistas para la aplicación Hernia.
Gestiona las solicitudes HTTP y coordina con los servicios.
"""

import logging
from io import BytesIO

import pytz
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.views import PasswordResetCompleteView
from django.core.paginator import Paginator
from django.db import transaction
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.cache import cache_control
from django.views.decorators.http import require_http_methods

from .forms import ImagenForm, RegistroForm, ProfileForm, UserForm
from .models import Imagen, Profile, Historial
from .services import (
    ImageProcessingService, 
    HistorialService, 
    ImageValidator
)
from .auth_service import AuthenticationService
from .pdf_service import InformeRadiologicoGenerator, HistorialGeneralGenerator
from .config import PAGINATION, ECUADOR_TIMEZONE

logger = logging.getLogger(__name__)


# ==================== AUTENTICACIÓN ====================
class CustomPasswordResetCompleteView(PasswordResetCompleteView):
    """Vista personalizada para completar reset de contraseña."""
    
    def get(self, request, *args, **kwargs):
        return redirect('login')


@cache_control(no_cache=True, must_revalidate=True, no_store=True)
@require_http_methods(["GET", "POST"])
def login_view(request):
    """
    Vista de login.
    Autentica usuarios por email y contraseña.
    """
    if request.method == 'POST':
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '')
        
        user, error = AuthenticationService.authenticate_by_email(email, password)
        
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            messages.error(request, error)
    
    return render(request, 'auth/login.html')


@require_http_methods(["GET", "POST"])
def register_view(request):
    """
    Vista de registro.
    Crea nuevos usuarios y los autentica automáticamente.
    """
    if request.method == 'POST':
        form = RegistroForm(request.POST)
        user, error = AuthenticationService.register_user(form)
        
        if user is not None:
            login(request, user)
            messages.success(request, '¡Registro exitoso! Has iniciado sesión automáticamente.')
            return redirect('index')
        else:
            messages.error(request, error)
    else:
        form = RegistroForm()
    
    return render(request, 'auth/register.html', {'form': form})


@cache_control(no_cache=True, must_revalidate=True, no_store=True)
@require_http_methods(["GET", "POST"])
def logout_view(request):
    """Vista de logout."""
    logout(request)
    return redirect('login')


@require_http_methods(["GET", "POST"])
def password_reset_view(request):
    """
    Vista de recuperación de contraseña.
    Envía un enlace de reset al email del usuario.
    """
    if request.method == 'POST':
        email = request.POST.get('email', '').strip()
        success, message = AuthenticationService.send_password_reset_email(email)
        
        if success:
            messages.success(request, message)
        else:
            messages.error(request, message)
        
        return redirect('password_reset')
    
    return render(request, 'auth/password_reset.html')


# ==================== VISTAS PRINCIPALES ====================
@login_required
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def index(request):
    """
    Vista principal (home).
    Muestra mensaje de bienvenida en la primera visita.
    """
    show_welcome_message = not request.session.get('welcome_message_shown', False)
    
    if show_welcome_message:
        request.session['welcome_message_shown'] = True
    
    return render(request, 'dashboard/home.html', {
        'user': request.user,
        'show_welcome_message': show_welcome_message,
    })


@login_required
@require_http_methods(["GET"])
def resultados(request):
    """Vista de resultados (plantilla base)."""
    return render(request, 'imagen/resultados.html')


@login_required
@require_http_methods(["GET"])
def ver_resultado(request, id):
    """
    Vista para ver un resultado específico.
    Solo permite ver resultados propios del usuario.
    """
    historial_item = get_object_or_404(Historial, id=id, user=request.user)
    
    ecuador_tz = pytz.timezone(ECUADOR_TIMEZONE)
    fecha_imagen_local = historial_item.fecha_imagen.astimezone(ecuador_tz)
    
    context = {
        'grupo': historial_item.grupo,
        'porcentaje': historial_item.porcentaje,
        'processed_image_url': historial_item.imagen.url,
        'fecha_imagen': fecha_imagen_local,
        'paciente_nombre': historial_item.paciente_nombre,
        'historial_id': historial_item.id
    }
    
    return render(request, 'imagen/resultados.html', context)


# ==================== PERFIL DE USUARIO ====================
@login_required
@require_http_methods(["GET", "POST"])
def profile_view(request):
    """
    Vista de perfil de usuario.
    Permite ver y editar información del perfil.
    """
    if request.method == 'POST':
        user_form = UserForm(request.POST, instance=request.user)
        profile_form = ProfileForm(request.POST, request.FILES, instance=request.user.profile)
        
        if user_form.is_valid() and profile_form.is_valid():
            with transaction.atomic():
                user_form.save()
                profile_form.save()
            messages.success(request, 'Perfil actualizado exitosamente.')
            return redirect('profile')
    else:
        user_form = UserForm(instance=request.user)
        profile_form = ProfileForm(instance=request.user.profile)
    
    return render(request, 'profile/perfil.html', {
        'user_form': user_form,
        'profile_form': profile_form
    })


@login_required
@require_http_methods(["GET", "POST"])
def editar_perfil(request):
    """
    Vista para editar perfil.
    Redirige a profile_view (evita duplicación).
    """
    return profile_view(request)


@login_required
@require_http_methods(["GET"])
def user_profile_view(request):
    """
    Vista de perfil de usuario (alternativa).
    Crea perfil si no existe.
    """
    user = request.user
    profile = getattr(user, 'profile', None)
    
    if profile is None:
        profile = Profile.objects.create(user=user)
    
    return render(request, 'profile/perfil.html', {'user': user, 'profile': profile})


# ==================== HISTORIAL MÉDICO ====================
@login_required
@require_http_methods(["GET"])
def historial_medico(request):
    """
    Vista del historial médico del usuario.
    Muestra solo los registros del usuario autenticado.
    """
    historial = Historial.objects.filter(user=request.user).order_by('-fecha_imagen')
    paginator = Paginator(historial, PAGINATION['ITEMS_PER_PAGE'])
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'historial/historial_medico.html', {'page_obj': page_obj})


@login_required
@require_http_methods(["GET"])
def historial_medico_general(request):
    """
    Vista del historial médico general.
    Solo accesible para superusuarios.
    """
    if not request.user.is_superuser:
        messages.error(request, 'No tienes permiso para acceder a esta sección.')
        return redirect('historial_med')
    
    historial = Historial.objects.all().order_by('-fecha_imagen')
    paginator = Paginator(historial, PAGINATION['ITEMS_PER_PAGE'])
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'historial/historial_medico_general.html', {'page_obj': page_obj})


@login_required
@require_http_methods(["POST"])
def eliminar_historial(request, id):
    """
    Elimina un historial del usuario.
    Solo permite eliminar registros propios.
    """
    try:
        success = HistorialService.delete_historial(id, user=request.user)
        if success:
            messages.success(request, 'El registro ha sido eliminado correctamente.')
        else:
            messages.error(request, 'No se encontró el registro.')
    except Exception as e:
        logger.error(f"Error eliminando historial: {str(e)}")
        messages.error(request, 'Error al eliminar el registro.')
    
    return redirect('historial_med')


@login_required
@require_http_methods(["POST"])
def eliminar_historial_general(request, id):
    """
    Elimina un historial (acceso general).
    Solo accesible para superusuarios.
    """
    if not request.user.is_superuser:
        messages.error(request, 'No tienes permiso para realizar esta acción.')
        return redirect('historial_med_gene')
    
    try:
        success = HistorialService.delete_historial(id)
        if success:
            messages.success(request, 'El registro ha sido eliminado correctamente.')
        else:
            messages.error(request, 'No se encontró el registro.')
    except Exception as e:
        logger.error(f"Error eliminando historial general: {str(e)}")
        messages.error(request, 'Error al eliminar el registro.')
    
    return redirect('historial_med_gene')


# ==================== PROCESAMIENTO DE IMÁGENES ====================
@login_required
@require_http_methods(["GET", "POST"])
def subir_imagen(request):
    """
    Vista para subir y procesar imágenes.
    Realiza inferencia y genera historial.
    """
    if request.method == 'POST':
        form = ImagenForm(request.POST, request.FILES)
        
        if not form.is_valid():
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, str(error))
            return render(request, 'imagen/subir_imagen.html', {'form': form})
        
        # Validar imagen
        imagen_file = request.FILES.get('imagen')
        is_valid, error_msg = ImageValidator.validate_image(imagen_file)
        
        if not is_valid:
            messages.error(request, error_msg)
            return render(request, 'imagen/subir_imagen.html', {'form': form})
        
        try:
            with transaction.atomic():
                # Procesar imagen
                imagen_obj = form.save(commit=False)
                paciente_nombre = form.cleaned_data.get('paciente_nombre')
                
                # Generar nombre encriptado
                encrypted_name = ImageProcessingService.generate_encrypted_filename(
                    imagen_file.name
                )
                imagen_obj.imagen.name = encrypted_name
                imagen_obj.paciente_nombre = paciente_nombre
                imagen_obj.save()
                
                # Descargar y procesar imagen
                image_url = imagen_obj.imagen.url
                image_pil = ImageProcessingService.download_image(image_url)
                
                if image_pil is None:
                    messages.error(request, 'Error al procesar la imagen.')
                    return render(request, 'imagen/subir_imagen.html', {'form': form})
                
                # Convertir a OpenCV
                import cv2
                import numpy as np
                img_cv2 = np.array(image_pil)
                img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
                
                # Realizar inferencia
                result = ImageProcessingService.get_inference_result(image_url)
                
                if result is None:
                    messages.error(request, 'Error al procesar la imagen con el modelo.')
                    return render(request, 'imagen/subir_imagen.html', {'form': form})
                
                # Dibujar predicciones
                predictions = result.get('predictions', [])
                img_cv2 = ImageProcessingService.draw_predictions_on_image(img_cv2, predictions)
                
                # Guardar imagen procesada
                buffer = ImageProcessingService.save_processed_image(img_cv2)
                imagen_obj.imagen.save(encrypted_name, buffer)
                
                # Extraer datos de predicción
                grupo, porcentaje = ImageProcessingService.extract_prediction_data(result)
                
                # Crear historial
                historial = HistorialService.create_historial_from_image(
                    request.user,
                    imagen_obj,
                    paciente_nombre,
                    grupo,
                    porcentaje
                )
                
                # Preparar contexto
                ecuador_tz = pytz.timezone(ECUADOR_TIMEZONE)
                fecha_local = imagen_obj.fecha.astimezone(ecuador_tz)
                
                context = {
                    'grupo': grupo,
                    'porcentaje': porcentaje,
                    'original_image_url': image_url,
                    'processed_image_url': imagen_obj.imagen.url,
                    'fecha_imagen': fecha_local,
                    'paciente_nombre': paciente_nombre
                }
                
                messages.success(request, 'Imagen procesada exitosamente.')
                return render(request, 'imagen/resultados.html', context)
        
        except Exception as e:
            logger.error(f"Error procesando imagen: {str(e)}")
            messages.error(request, 'Error al procesar la imagen. Intenta nuevamente.')
            return render(request, 'imagen/subir_imagen.html', {'form': form})
    
    else:
        form = ImagenForm()
    
    return render(request, 'imagen/subir_imagen.html', {'form': form})


# ==================== GENERACIÓN DE PDFs ====================
@login_required
@require_http_methods(["GET"])
def generar_pdf_fila(request, id):
    """
    Genera un PDF con el informe radiológico de un historial.
    Solo permite generar PDFs de registros propios.
    """
    try:
        historial = get_object_or_404(Historial, id=id, user=request.user)
        
        generator = InformeRadiologicoGenerator()
        pdf = generator.generate(historial)
        
        response = HttpResponse(pdf, content_type='application/pdf')
        filename = f"informe_rad_{str(historial.id).zfill(6)}_{historial.paciente_nombre.replace(' ', '_')}.pdf"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        logger.info(f"PDF generado: {filename}")
        return response
    
    except Exception as e:
        logger.error(f"Error generando PDF individual: {str(e)}")
        messages.error(request, 'Error al generar el PDF.')
        return redirect('historial_med')


@login_required
@require_http_methods(["GET"])
def generar_pdf_general(request):
    """
    Genera un PDF con el historial general.
    Muestra todos los registros si es superusuario, solo los propios si no.
    """
    try:
        if request.user.is_superuser:
            historiales = Historial.objects.all().order_by('-fecha_imagen')
            filename = 'historial_radiologico_completo.pdf'
        else:
            historiales = Historial.objects.filter(user=request.user).order_by('-fecha_imagen')
            filename = f'historial_rad_{request.user.username}.pdf'
        
        generator = HistorialGeneralGenerator()
        pdf = generator.generate(historiales)
        
        response = HttpResponse(pdf, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        logger.info(f"PDF general generado: {filename}")
        return response
    
    except Exception as e:
        logger.error(f"Error generando PDF general: {str(e)}")
        messages.error(request, 'Error al generar el PDF.')
        return redirect('historial_med')

