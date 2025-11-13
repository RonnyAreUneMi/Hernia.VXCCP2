# ==================== SERVICIO DE GENERACIÓN DE PDFs ====================
"""
Módulo para generar reportes en PDF.
Centraliza la lógica de diseño y generación de PDFs.
"""

import os
import tempfile
import logging
from io import BytesIO
from typing import Optional

import pytz
import requests
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

logger = logging.getLogger(__name__)


class PDFStyles:
    """Estilos y colores para PDFs."""
    
    # Colores
    AZUL_OSCURO = HexColor('#1a2332')
    AZUL_MEDIO = HexColor('#2c3e50')
    GRIS_TEXTO = HexColor('#2d3748')
    GRIS_LINEA = HexColor('#cbd5e0')
    VERDE_CLINICO = HexColor('#059669')
    ROJO_CLINICO = HexColor('#dc2626')
    FONDO_CLARO = HexColor('#f8fafc')
    GRIS_CLARO = HexColor('#64748b')
    GRIS_MUY_CLARO = HexColor('#94a3b8')
    
    # Dimensiones
    MARGIN_H = 0.6 * inch
    MARGIN_V = 0.75 * inch
    HEADER_HEIGHT = 0.9 * inch
    FOOTER_HEIGHT = 1 * inch


class PDFGenerator:
    """Generador base de PDFs."""
    
    def __init__(self):
        self.width, self.height = letter
        self.buffer = BytesIO()
        self.p = canvas.Canvas(self.buffer, pagesize=letter)
    
    def draw_header(self, title: str, subtitle: str = "", right_text: str = ""):
        """Dibuja el encabezado del PDF."""
        self.p.setFillColor(PDFStyles.AZUL_OSCURO)
        self.p.rect(0, self.height - PDFStyles.HEADER_HEIGHT, self.width, PDFStyles.HEADER_HEIGHT, fill=1, stroke=0)
        
        self.p.setFillColor(colors.white)
        self.p.setFont("Helvetica-Bold", 18)
        self.p.drawString(PDFStyles.MARGIN_H, self.height - 0.45*inch, title)
        
        if subtitle:
            self.p.setFont("Helvetica", 9)
            self.p.drawString(PDFStyles.MARGIN_H, self.height - 0.65*inch, subtitle)
        
        if right_text:
            self.p.setFont("Helvetica", 8)
            self.p.drawRightString(self.width - PDFStyles.MARGIN_H, self.height - 0.45*inch, right_text)
        
        self.p.setStrokeColor(PDFStyles.GRIS_LINEA)
        self.p.setLineWidth(0.5)
        self.p.line(PDFStyles.MARGIN_H, self.height - PDFStyles.HEADER_HEIGHT - 0.05*inch, 
                   self.width - PDFStyles.MARGIN_H, self.height - PDFStyles.HEADER_HEIGHT - 0.05*inch)
    
    def draw_footer(self, text: str = ""):
        """Dibuja el pie de página."""
        self.p.setStrokeColor(PDFStyles.GRIS_LINEA)
        self.p.setLineWidth(0.5)
        self.p.line(PDFStyles.MARGIN_H, 1*inch, self.width - PDFStyles.MARGIN_H, 1*inch)
        
        self.p.setFillColor(PDFStyles.GRIS_CLARO)
        self.p.setFont("Helvetica", 7)
        self.p.drawString(PDFStyles.MARGIN_H, 0.75*inch, "NOTA IMPORTANTE:")
        self.p.setFont("Helvetica", 6.5)
        self.p.drawString(PDFStyles.MARGIN_H, 0.6*inch, 
                         "Este informe ha sido generado mediante análisis automatizado con inteligencia artificial y debe ser validado por un médico radiólogo certificado.")
        self.p.drawString(PDFStyles.MARGIN_H, 0.47*inch, 
                         "Los resultados deben interpretarse en el contexto clínico del paciente. No sustituye el criterio médico profesional.")
    
    def draw_section_box(self, x: float, y: float, width: float, height: float, 
                        title: str, content_func=None) -> float:
        """Dibuja una caja de sección con título."""
        self.p.setFillColor(colors.white)
        self.p.rect(x, y - height, width, height, fill=1, stroke=0)
        
        self.p.setStrokeColor(PDFStyles.GRIS_LINEA)
        self.p.setLineWidth(0.5)
        self.p.rect(x, y - height, width, height, fill=0, stroke=1)
        
        self.p.setFillColor(PDFStyles.AZUL_MEDIO)
        self.p.setFont("Helvetica-Bold", 9)
        self.p.drawString(x + 0.15*inch, y - 0.25*inch, title)
        
        self.p.setStrokeColor(PDFStyles.AZUL_MEDIO)
        self.p.setLineWidth(1.5)
        self.p.line(x + 0.15*inch, y - 0.35*inch, x + 1.5*inch, y - 0.35*inch)
        
        if content_func:
            content_func(x, y)
        
        return y - height
    
    def get_pdf(self) -> bytes:
        """Obtiene el PDF generado."""
        self.p.save()
        pdf = self.buffer.getvalue()
        self.buffer.close()
        return pdf


class InformeRadiologicoGenerator(PDFGenerator):
    """Generador de informes radiológicos individuales."""
    
    def generate(self, historial) -> bytes:
        """Genera un informe radiológico completo."""
        ecuador_tz = pytz.timezone('America/Guayaquil')
        fecha_local = historial.fecha_imagen.astimezone(ecuador_tz)
        
        # Encabezado
        self.draw_header(
            "INFORME RADIOLÓGICO",
            "Departamento de Diagnóstico por Imagen",
            f"No. {str(historial.id).zfill(6)}"
        )
        
        # Imagen radiológica
        self._draw_radiographic_image(historial, fecha_local)
        
        # Información clínica (lado derecho)
        self._draw_clinical_info(historial, fecha_local)
        
        # Pie de página
        self.draw_footer()
        
        self.p.showPage()
        return self.get_pdf()
    
    def _draw_radiographic_image(self, historial, fecha_local):
        """Dibuja la imagen radiológica."""
        img_x = PDFStyles.MARGIN_H
        img_y = self.height - 9.8*inch
        img_width = 4.2*inch
        img_height = 7.8*inch
        
        self.p.setStrokeColor(PDFStyles.GRIS_LINEA)
        self.p.setLineWidth(1)
        self.p.rect(img_x, img_y, img_width, img_height, stroke=1, fill=0)
        
        if historial.imagen:
            try:
                image_url = historial.imagen.url
                response = requests.get(image_url, timeout=10)
                
                if response.status_code == 200 and response.content:
                    image_data = BytesIO(response.content)
                    img = Image.open(image_data).convert('RGB')
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                        img.save(temp_file, format='JPEG', quality=95)
                        temp_file_path = temp_file.name
                    
                    self.p.drawImage(
                        temp_file_path, 
                        img_x + 0.05*inch, 
                        img_y + 0.05*inch,
                        width=img_width - 0.1*inch, 
                        height=img_height - 0.1*inch,
                        preserveAspectRatio=True, 
                        mask='auto'
                    )
                    os.remove(temp_file_path)
            except Exception as e:
                logger.error(f"Error dibujando imagen en PDF: {str(e)}")
                self.p.setFillColor(PDFStyles.GRIS_TEXTO)
                self.p.setFont("Helvetica", 9)
                self.p.drawCentredString(img_x + img_width/2, img_y + img_height/2, "Imagen no disponible")
        
        self.p.setFillColor(PDFStyles.AZUL_MEDIO)
        self.p.setFont("Helvetica", 8)
        self.p.drawString(img_x, img_y - 0.25*inch, "Fig. 1 - Radiografía de tórax con marcación automatizada")
    
    def _draw_clinical_info(self, historial, fecha_local):
        """Dibuja la información clínica."""
        right_x = 5.1*inch
        y_pos = self.height - 1.3*inch
        box_width = 2.9*inch
        
        # Datos del paciente
        y_pos = self._draw_patient_data(historial, fecha_local, right_x, y_pos, box_width)
        
        # Hallazgos
        y_pos = self._draw_findings(historial, right_x, y_pos, box_width)
        
        # Índice de confianza
        y_pos = self._draw_confidence_index(historial, right_x, y_pos, box_width)
        
        # Interpretación radiológica
        self._draw_radiological_interpretation(historial, right_x, y_pos, box_width)
    
    def _draw_patient_data(self, historial, fecha_local, x, y, width):
        """Dibuja datos del paciente."""
        self.p.setFillColor(PDFStyles.FONDO_CLARO)
        self.p.rect(x, y - 1.35*inch, width, 1.35*inch, fill=1, stroke=0)
        
        self.p.setStrokeColor(PDFStyles.GRIS_LINEA)
        self.p.setLineWidth(0.5)
        self.p.rect(x, y - 1.35*inch, width, 1.35*inch, fill=0, stroke=1)
        
        self.p.setFillColor(PDFStyles.AZUL_MEDIO)
        self.p.setFont("Helvetica-Bold", 9)
        self.p.drawString(x + 0.15*inch, y - 0.25*inch, "DATOS DEL PACIENTE")
        
        self.p.setStrokeColor(PDFStyles.AZUL_MEDIO)
        self.p.setLineWidth(1.5)
        self.p.line(x + 0.15*inch, y - 0.35*inch, x + 1.5*inch, y - 0.35*inch)
        
        self.p.setFillColor(PDFStyles.GRIS_TEXTO)
        self.p.setFont("Helvetica-Bold", 8)
        self.p.drawString(x + 0.15*inch, y - 0.55*inch, "Paciente:")
        self.p.setFont("Helvetica", 8)
        self.p.drawString(x + 0.15*inch, y - 0.7*inch, historial.paciente_nombre)
        
        self.p.setFont("Helvetica-Bold", 8)
        self.p.drawString(x + 0.15*inch, y - 0.9*inch, "Médico solicitante:")
        self.p.setFont("Helvetica", 8)
        self.p.drawString(x + 0.15*inch, y - 1.05*inch, historial.user.username)
        
        self.p.setFont("Helvetica", 7)
        self.p.setFillColor(PDFStyles.GRIS_CLARO)
        self.p.drawString(
            x + 0.15*inch, y - 1.25*inch,
            f"Fecha: {fecha_local.strftime('%d/%m/%Y')} | Hora: {fecha_local.strftime('%H:%M')}"
        )
        
        return y - 1.65*inch
    
    def _draw_findings(self, historial, x, y, width):
        """Dibuja hallazgos."""
        self.p.setFillColor(colors.white)
        self.p.rect(x, y - 1*inch, width, 1*inch, fill=1, stroke=0)
        
        self.p.setStrokeColor(PDFStyles.GRIS_LINEA)
        self.p.setLineWidth(0.5)
        self.p.rect(x, y - 1*inch, width, 1*inch, fill=0, stroke=1)
        
        self.p.setFillColor(PDFStyles.AZUL_MEDIO)
        self.p.setFont("Helvetica-Bold", 9)
        self.p.drawString(x + 0.15*inch, y - 0.25*inch, "HALLAZGOS")
        
        self.p.setStrokeColor(PDFStyles.AZUL_MEDIO)
        self.p.setLineWidth(1.5)
        self.p.line(x + 0.15*inch, y - 0.35*inch, x + 1.1*inch, y - 0.35*inch)
        
        diagnostico_color = PDFStyles.VERDE_CLINICO if historial.grupo == "Sin Hernia" else PDFStyles.ROJO_CLINICO
        
        self.p.setFillColor(diagnostico_color)
        self.p.circle(x + 0.25*inch, y - 0.57*inch, 0.08*inch, fill=1, stroke=0)
        
        self.p.setFillColor(PDFStyles.GRIS_TEXTO)
        self.p.setFont("Helvetica-Bold", 11)
        self.p.drawString(x + 0.45*inch, y - 0.62*inch, historial.grupo.upper())
        
        self.p.setFont("Helvetica", 7)
        self.p.setFillColor(PDFStyles.GRIS_CLARO)
        self.p.drawString(x + 0.45*inch, y - 0.78*inch, f"Confiabilidad del análisis: {historial.porcentaje}%")
        
        # Mostrar ubicación si está disponible
        if hasattr(historial, 'ubicacion') and historial.ubicacion and historial.ubicacion != 'No especificada':
            self.p.setFont("Helvetica", 7)
            self.p.setFillColor(PDFStyles.GRIS_TEXTO)
            self.p.drawString(x + 0.45*inch, y - 0.92*inch, historial.ubicacion)
        
        return y - 1.2*inch
    
    def _draw_confidence_index(self, historial, x, y, width):
        """Dibuja índice de confianza."""
        self.p.setFillColor(PDFStyles.FONDO_CLARO)
        self.p.rect(x, y - 0.85*inch, width, 0.85*inch, fill=1, stroke=0)
        
        self.p.setStrokeColor(PDFStyles.GRIS_LINEA)
        self.p.setLineWidth(0.5)
        self.p.rect(x, y - 0.85*inch, width, 0.85*inch, fill=0, stroke=1)
        
        self.p.setFillColor(PDFStyles.AZUL_MEDIO)
        self.p.setFont("Helvetica-Bold", 9)
        self.p.drawString(x + 0.15*inch, y - 0.25*inch, "ÍNDICE DE CONFIANZA")
        
        self.p.setStrokeColor(PDFStyles.AZUL_MEDIO)
        self.p.setLineWidth(1.5)
        self.p.line(x + 0.15*inch, y - 0.35*inch, x + 1.6*inch, y - 0.35*inch)
        
        bar_x = x + 0.15*inch
        bar_y = y - 0.55*inch
        bar_width = width - 0.3*inch
        bar_height = 0.12*inch
        
        self.p.setFillColor(HexColor('#e2e8f0'))
        self.p.rect(bar_x, bar_y, bar_width, bar_height, fill=1, stroke=0)
        
        confidence_fill = bar_width * (float(historial.porcentaje) / 100)
        self.p.setFillColor(PDFStyles.AZUL_MEDIO)
        self.p.rect(bar_x, bar_y, confidence_fill, bar_height, fill=1, stroke=0)
        
        self.p.setFillColor(PDFStyles.AZUL_MEDIO)
        self.p.setFont("Helvetica-Bold", 16)
        self.p.drawRightString(x + width - 0.15*inch, y - 0.75*inch, f"{historial.porcentaje}%")
        
        return y - 1.05*inch
    
    def _draw_radiological_interpretation(self, historial, x, y, width):
        """Dibuja interpretación radiológica."""
        self.p.setFillColor(colors.white)
        self.p.rect(x, y - 2.6*inch, width, 2.6*inch, fill=1, stroke=0)
        
        self.p.setStrokeColor(PDFStyles.GRIS_LINEA)
        self.p.setLineWidth(0.5)
        self.p.rect(x, y - 2.6*inch, width, 2.6*inch, fill=0, stroke=1)
        
        self.p.setFillColor(PDFStyles.AZUL_MEDIO)
        self.p.setFont("Helvetica-Bold", 9)
        self.p.drawString(x + 0.15*inch, y - 0.25*inch, "INTERPRETACIÓN RADIOLÓGICA")
        
        self.p.setStrokeColor(PDFStyles.AZUL_MEDIO)
        self.p.setLineWidth(1.5)
        self.p.line(x + 0.15*inch, y - 0.35*inch, x + 2.1*inch, y - 0.35*inch)
        
        self.p.setFillColor(PDFStyles.GRIS_TEXTO)
        self.p.setFont("Helvetica", 7.5)
        
        text_y = y - 0.55*inch
        line_height = 0.14*inch
        
        if historial.grupo == "Sin Hernia":
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
                self.p.setFont("Helvetica-Bold", 7.5)
            elif linea.startswith("•"):
                self.p.setFont("Helvetica", 7)
            else:
                self.p.setFont("Helvetica", 7.5)
            
            self.p.drawString(x + 0.15*inch, text_y, linea)
            text_y -= line_height


class HistorialGeneralGenerator(PDFGenerator):
    """Generador de historial general en PDF."""
    
    def generate(self, historiales) -> bytes:
        """Genera un PDF con múltiples historiales."""
        ecuador_tz = pytz.timezone('America/Guayaquil')
        
        for index, item in enumerate(historiales):
            if index > 0:
                self.p.showPage()
            
            fecha_local = item.fecha_imagen.astimezone(ecuador_tz)
            
            self._draw_page(item, fecha_local, index + 1, len(historiales))
        
        self.p.save()
        return self.get_pdf()
    
    def _draw_page(self, item, fecha_local, page_num, total_pages):
        """Dibuja una página del historial."""
        self.p.setFillColor(PDFStyles.AZUL_OSCURO)
        self.p.rect(0, self.height - 0.9*inch, self.width, 0.9*inch, fill=1, stroke=0)
        
        self.p.setFillColor(colors.white)
        self.p.setFont("Helvetica-Bold", 16)
        self.p.drawString(PDFStyles.MARGIN_H, self.height - 0.5*inch, "HISTORIAL RADIOLÓGICO")
        self.p.setFont("Helvetica", 8)
        self.p.drawRightString(self.width - PDFStyles.MARGIN_H, self.height - 0.5*inch, 
                              f"Registro {page_num} de {total_pages}")
        
        y_pos = self.height - 1.3*inch
        
        # Tabla de información
        ubicacion_texto = getattr(item, 'ubicacion', 'No especificada')
        if ubicacion_texto == 'No especificada':
            ubicacion_texto = 'N/A'
        
        data = [
            ["Paciente:", item.paciente_nombre, "ID:", str(item.id).zfill(6)],
            ["Médico:", item.user.username, "Fecha:", fecha_local.strftime('%d/%m/%Y %H:%M')],
            ["Diagnóstico:", item.grupo, "Confianza:", f"{item.porcentaje}%"],
            ["Ubicación:", ubicacion_texto, "", ""],
        ]
        
        table = Table(data, colWidths=[1*inch, 2.5*inch, 0.9*inch, 2.1*inch])
        table.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 8),
            ('FONT', (2, 0), (2, -1), 'Helvetica-Bold', 8),
            ('FONT', (1, 0), (1, -1), 'Helvetica', 8),
            ('FONT', (3, 0), (3, -1), 'Helvetica', 8),
            ('TEXTCOLOR', (0, 0), (-1, -1), PDFStyles.GRIS_TEXTO),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LINEABOVE', (0, 0), (-1, 0), 0.5, PDFStyles.GRIS_LINEA),
            ('LINEBELOW', (0, -1), (-1, -1), 0.5, PDFStyles.GRIS_LINEA),
        ]))
        
        table.wrapOn(self.p, self.width, self.height)
        table.drawOn(self.p, PDFStyles.MARGIN_H, y_pos - 0.8*inch)
        
        y_pos -= 1.4*inch
        
        # Imagen
        if item.imagen:
            img_width = 3.5*inch
            img_height = 5.5*inch
            margin_left = (self.width - img_width) / 2
            
            self.p.setStrokeColor(PDFStyles.GRIS_LINEA)
            self.p.setLineWidth(1)
            self.p.rect(margin_left, y_pos - img_height, img_width, img_height, stroke=1, fill=0)
            
            try:
                image_url = item.imagen.url
                response = requests.get(image_url, timeout=10)
                
                if response.status_code == 200 and response.content:
                    image_data = BytesIO(response.content)
                    img = Image.open(image_data).convert('RGB')
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                        img.save(temp_file, format='JPEG', quality=95)
                        temp_file_path = temp_file.name
                    
                    self.p.drawImage(
                        temp_file_path, 
                        margin_left + 0.05*inch, 
                        y_pos - img_height + 0.05*inch,
                        width=img_width - 0.1*inch, 
                        height=img_height - 0.1*inch,
                        preserveAspectRatio=True, 
                        mask='auto'
                    )
                    os.remove(temp_file_path)
            except Exception as e:
                logger.error(f"Error dibujando imagen en historial PDF: {str(e)}")
        
        self.p.setFillColor(PDFStyles.GRIS_MUY_CLARO)
        self.p.setFont("Helvetica", 7)
        self.p.drawCentredString(self.width/2, 0.4*inch, 
                                f"Generado: {fecha_local.strftime('%d/%m/%Y %H:%M:%S')}")
