# ==================== SERVICIOS DE PROCESAMIENTO ====================
"""
Módulo de servicios para procesamiento de imágenes y generación de reportes.
Contiene la lógica de negocio separada de las vistas.
"""

import os
import hashlib
import tempfile
import logging
from io import BytesIO
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import pytz
import requests
from PIL import Image
from inference_sdk import InferenceHTTPClient

from django.conf import settings
from django.core.files.base import ContentFile
from django.db import transaction

from .models import Imagen, Historial

from ultralytics import YOLO


logger = logging.getLogger(__name__)

# ==================== CONFIGURACIÓN ====================
# def get_inference_client():
#     """Obtiene el cliente de inferencia configurado."""
#     api_key = settings.ROBOFLOW_API_KEY
#     if not api_key:
#         raise ValueError("ROBOFLOW_API_KEY no está configurada en settings")
    
#     return InferenceHTTPClient(
#         api_url="https://outline.roboflow.com",
#         api_key=api_key
#     )


# ==================== SERVICIOS DE IMAGEN ====================
class ImageProcessingService:
    """Servicio para procesar imágenes y realizar inferencias."""
    
    ECUADOR_TZ = pytz.timezone('America/Guayaquil')
    MODEL_ID = "proy_2/1"
    CONFIDENCE_THRESHOLD = 0.0
    
    @staticmethod
    def generate_encrypted_filename(original_name: str) -> str:
        """Genera un nombre de archivo encriptado."""
        hash_object = hashlib.sha256(original_name.encode())
        encrypted_name = hash_object.hexdigest()
        extension = original_name.split('.')[-1]
        return f"{encrypted_name}.{extension}"
    
    @staticmethod
    def download_image(image_url: str) -> Optional[Image.Image]:
        """Descarga una imagen desde URL."""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            if not response.content:
                logger.warning(f"Respuesta vacía al descargar imagen: {image_url}")
                return None
            
            image_data = BytesIO(response.content)
            return Image.open(image_data).convert('RGB')
        except requests.RequestException as e:
            logger.error(f"Error descargando imagen {image_url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error procesando imagen descargada: {str(e)}")
            return None
    
    @staticmethod
    def get_inference_result(image_url: str) -> Optional[Dict]:
        """
        Obtiene el resultado de inferencia usando el modelo YOLO local.
        """
        try:
            model_path = os.path.join(settings.BASE_DIR, 'app', 'Hernia', 'yolo', 'best.pt')

            if not os.path.exists(model_path):
                logger.error(f"❌ No se encontró el modelo YOLO en: {model_path}")
                return None

            if not hasattr(ImageProcessingService, '_model'):
                ImageProcessingService._model = YOLO(model_path)

            model = ImageProcessingService._model

            if image_url.startswith('http'):
                import requests, tempfile
                response = requests.get(image_url, stream=True)
                response.raw.decode_content = True
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                with open(temp_file.name, 'wb') as f:
                    f.write(response.content)
                image_path = temp_file.name
            else:
                image_path = os.path.join(settings.BASE_DIR, image_url.lstrip('/'))

            results = model(image_path)

            predictions = []
            for box in results[0].boxes:
                predictions.append({
                    'class': model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                })

            return {'predictions': predictions}

        except Exception as e:
            logger.error(f"Error en inferencia con YOLO: {str(e)}")
            return None

    
    @staticmethod
    def draw_predictions_on_image(img_cv2: np.ndarray, predictions: list) -> np.ndarray:
        output = img_cv2.copy()
        img_height, img_width = img_cv2.shape[:2]
        
        colors = {
            'l1': (0, 165, 255),      # Naranja
            'l2': (0, 200, 255),      # Amarillo-naranja
            'l3': (0, 255, 200),      # Amarillo-verde
            'l4': (100, 255, 100),    # Verde claro
            'l5': (200, 255, 0),      # Cyan
            's1': (255, 150, 100),    # Azul claro
            'hernia': (0, 0, 255),    # Rojo
            'sin hernia': (0, 255, 0) # Verde
        }
        
        def hay_overlap(box1, box2, threshold=0.1):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            
            x_left = max(x1_min, x2_min)
            y_top = max(y1_min, y2_min)
            x_right = min(x1_max, x2_max)
            y_bottom = min(y1_max, y2_max)
            
            if x_right < x_left or y_bottom < y_top:
                return False
            
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            
            iou = intersection_area / min(box1_area, box2_area)
            return iou > threshold
        
        detections = []
        for pred in predictions:
            try:
                x1, y1, x2, y2 = map(int, pred.get('bbox', [0, 0, 0, 0]))
                confidence = pred.get('confidence', 0) * 100
                class_name = pred.get('class', 'Desconocido')
                detections.append({
                    'box': (x1, y1, x2, y2),
                    'class': class_name,
                    'confidence': confidence
                })
            except Exception as e:
                logger.warning(f"Error procesando predicción: {str(e)}")
                continue
        
        hernias = [d for d in detections if 'hernia' in d['class'].lower() and 'sin' not in d['class'].lower()]
        todas_vertebras = [d for d in detections if d['class'].lower() in ['l1', 'l2', 'l3', 'l4', 'l5', 's1']]
        vertebras_con_hernia = []
        for vertebra in todas_vertebras:
            for hernia in hernias:
                if hay_overlap(vertebra['box'], hernia['box']):
                    vertebras_con_hernia.append(vertebra)
                    break
        
        detecciones_a_mostrar = hernias + vertebras_con_hernia
        
        for det in detecciones_a_mostrar:
            try:
                x1, y1, x2, y2 = det['box']
                class_name = det['class']
                class_name_lower = class_name.lower()
                confidence = det['confidence']
                
                color = colors.get(class_name_lower, (255, 255, 255))
                
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2
                
                overlay = output.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
                
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name} {confidence:.2f}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )
                
                if 'hernia' in class_name_lower and 'sin' not in class_name_lower:
                    label_x = x2 + 40
                    line_start = (x2, y_center)
                    line_end = (label_x - 10, y_center)
                else:
                    label_x = x1 - text_width - 50
                    line_start = (x1, y_center)
                    line_end = (label_x + text_width + 10, y_center)
                
                label_y = y_center + text_height // 2
                
                if label_x < 0:
                    label_x = 10
                if label_x + text_width > img_width:
                    label_x = img_width - text_width - 10
                
                cv2.line(output, line_start, line_end, color, 2)
                
                cv2.circle(output, line_start, 4, color, -1)
                
                padding = 5
                cv2.rectangle(output,
                            (label_x - padding, label_y - text_height - padding),
                            (label_x + text_width + padding, label_y + baseline + padding),
                            color, -1)
                
                cv2.rectangle(output,
                            (label_x - padding, label_y - text_height - padding),
                            (label_x + text_width + padding, label_y + baseline + padding),
                            (0, 0, 0), 2)
                
                cv2.putText(output, label, (label_x, label_y),
                        font, font_scale, (0, 0, 0), font_thickness)
                
            except Exception as e:
                logger.warning(f"Error dibujando predicción: {str(e)}")
                continue
        
        return output

    
    @staticmethod
    def save_processed_image(img_cv2: np.ndarray) -> BytesIO:
        """Convierte imagen procesada a BytesIO."""
        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        img_pil.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        return buffer
    
    @staticmethod
    def extract_prediction_data(result: Dict) -> Tuple[str, float]:
        """Extrae datos de predicción del resultado."""
        predictions = result.get('predictions', [])
        
        if not predictions:
            return "Predicción no encontrada", 0.0
        
        first_pred = predictions[0]
        class_prediction = first_pred.get('class', 'Unknown')
        confidence = first_pred.get('confidence', 0) * 100
        
        grupo = "Sin Hernia" if class_prediction == 'Sin Hernia' else "Hernia"
        porcentaje = round(confidence, 2)
        
        return grupo, porcentaje


class HistorialService:
    """Servicio para gestionar el historial médico."""
    
    ECUADOR_TZ = pytz.timezone('America/Guayaquil')
    
    @staticmethod
    @transaction.atomic
    def create_historial_from_image(
        user,
        imagen_obj: Imagen,
        paciente_nombre: str,
        grupo: str,
        porcentaje: float
    ) -> Historial:
        """Crea un registro de historial de forma atómica."""
        try:
            fecha_local = imagen_obj.fecha.astimezone(HistorialService.ECUADOR_TZ)
            
            historial = Historial(
                user=user,
                imagen=imagen_obj.imagen,
                porcentaje=porcentaje,
                grupo=grupo,
                paciente_nombre=paciente_nombre,
                fecha_imagen=fecha_local,
            )
            historial.full_clean()
            historial.save()
            
            logger.info(
                f"Historial creado: ID={historial.id}, "
                f"Paciente={paciente_nombre}, Grupo={grupo}"
            )
            return historial
        except Exception as e:
            logger.error(f"Error creando historial: {str(e)}")
            raise
    
    @staticmethod
    @transaction.atomic
    def delete_historial(historial_id: int, user=None) -> bool:
        """Elimina un historial de forma atómica."""
        try:
            if user:
                historial = Historial.objects.get(id=historial_id, user=user)
            else:
                historial = Historial.objects.get(id=historial_id)
            
            if historial.imagen:
                historial.imagen.delete()
            
            historial.delete()
            logger.info(f"Historial eliminado: ID={historial_id}")
            return True
        except Historial.DoesNotExist:
            logger.warning(f"Historial no encontrado: ID={historial_id}")
            return False
        except Exception as e:
            logger.error(f"Error eliminando historial: {str(e)}")
            raise


# ==================== VALIDADORES ====================
class ImageValidator:
    """Validador para imágenes subidas."""
    
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    MIN_DIMENSION = 100
    MAX_DIMENSION = 10000
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """Valida la extensión del archivo."""
        extension = filename.rsplit('.', 1)[-1].lower()
        return extension in ImageValidator.ALLOWED_EXTENSIONS
    
    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """Valida el tamaño del archivo."""
        return 0 < file_size <= ImageValidator.MAX_FILE_SIZE
    
    @staticmethod
    def validate_image_dimensions(image: Image.Image) -> bool:
        """Valida las dimensiones de la imagen."""
        width, height = image.size
        return (
            ImageValidator.MIN_DIMENSION <= width <= ImageValidator.MAX_DIMENSION and
            ImageValidator.MIN_DIMENSION <= height <= ImageValidator.MAX_DIMENSION
        )
    
    @staticmethod
    def validate_image(file_obj) -> Tuple[bool, Optional[str]]:
        """Valida una imagen completa."""
        try:
            # Validar extensión
            if not ImageValidator.validate_file_extension(file_obj.name):
                return False, "Formato de archivo no permitido"
            
            # Validar tamaño
            if not ImageValidator.validate_file_size(file_obj.size):
                return False, f"Archivo demasiado grande (máximo {ImageValidator.MAX_FILE_SIZE / 1024 / 1024}MB)"
            
            # Validar que sea una imagen válida
            try:
                image = Image.open(file_obj)
                image.verify()
                file_obj.seek(0)
                image = Image.open(file_obj)
            except Exception:
                return False, "Archivo no es una imagen válida"
            
            # Validar dimensiones
            if not ImageValidator.validate_image_dimensions(image):
                return False, f"Dimensiones inválidas ({ImageValidator.MIN_DIMENSION}-{ImageValidator.MAX_DIMENSION}px)"
            
            return True, None
        except Exception as e:
            logger.error(f"Error validando imagen: {str(e)}")
            return False, "Error al validar la imagen"

