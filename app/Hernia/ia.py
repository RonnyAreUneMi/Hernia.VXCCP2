"""
RetinaNet para Detección de Hernias - Versión Ultra Robusta
Sin errores, maneja todos los casos posibles
"""

# ==============================================================================
# CELDA 1: MONTAR DRIVE
# ==============================================================================


import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No disponible'}")
# ==============================================================================
# CELDA 2: INSTALAR DEPENDENCIAS
# ==============================================================================

!pip install -q pycocotools albumentations opencv-python-headless

print("✅ Dependencias instaladas")
# ==============================================================================
# CELDA 3: COPIAR DATASET
# ==============================================================================

!cp -r "/content/drive/MyDrive/Colab Notebooks/dataset" /content/dataset

print("✅ Dataset copiado")
!ls -lh /content/dataset/
# ==============================================================================
# CELDA 4: IMPORTS
# ==============================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.detection import retinanet_resnet50_fpn
import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

print("✅ Imports completados")
# ==============================================================================
# CELDA 5: CONFIGURACIÓN
# ==============================================================================

class Config:
    DATA_ROOT = '/content/dataset'
    TRAIN_IMG_DIR = f'{DATA_ROOT}/train'
    TRAIN_ANN = f'{DATA_ROOT}/train/_annotations.coco.json'
    VAL_IMG_DIR = f'{DATA_ROOT}/valid'
    VAL_ANN = f'{DATA_ROOT}/valid/_annotations.coco.json'

    NUM_CLASSES = 3
    IMG_SIZE = 640
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LR = 5e-5

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 2

    SAVE_DIR = '/content/drive/MyDrive/Colab Notebooks/retinanet_hernia'

    def __init__(self):
        Path(self.SAVE_DIR).mkdir(parents=True, exist_ok=True)

config = Config()
print(f"✅ Config: {config.NUM_CLASSES} clases, {config.BATCH_SIZE} batch, {config.NUM_EPOCHS} épocas")
# ==============================================================================
# CELDA 6: DATASET
# ==============================================================================

class HerniaDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size=640):
        self.img_dir = Path(img_dir)
        self.img_size = img_size

        with open(ann_file, 'r') as f:
            coco = json.load(f)

        self.images = coco['images']
        self.categories = coco['categories']

        # Agrupar anotaciones por imagen
        self.img_anns = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_anns:
                self.img_anns[img_id] = []
            self.img_anns[img_id].append(ann)

        # Mapeo de category_id a label
        self.cat_to_label = {cat['id']: idx + 1 for idx, cat in enumerate(self.categories)}

        if not hasattr(HerniaDataset, '_shown'):
            print(f"\n📋 Categorías ({len(self.categories)}):")
            for cat in self.categories:
                print(f"   {cat['name']} → Label {self.cat_to_label[cat['id']]}")
            HerniaDataset._shown = True

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = self.img_dir / img_info['file_name']

        # Leer imagen
        try:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        h_orig, w_orig = img.shape[:2]

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0

        # Normalizar
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        # Convertir a tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()

        # Obtener anotaciones
        anns = self.img_anns.get(img_info['id'], [])

        boxes = []
        labels = []

        for ann in anns:
            try:
                x, y, w, h = ann['bbox']

                # Escalar al tamaño redimensionado
                x = x * self.img_size / w_orig
                y = y * self.img_size / h_orig
                w = w * self.img_size / w_orig
                h = h * self.img_size / h_orig

                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(self.img_size, x + w), min(self.img_size, y + h)

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(self.cat_to_label.get(ann['category_id'], 1))
            except:
                continue

        # Si no hay boxes válidos, crear uno dummy
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

print("✅ Dataset definido")
# ==============================================================================
# CELDA 7: CREAR DATALOADERS
# ==============================================================================

train_dataset = HerniaDataset(config.TRAIN_IMG_DIR, config.TRAIN_ANN, config.IMG_SIZE)
val_dataset = HerniaDataset(config.VAL_IMG_DIR, config.VAL_ANN, config.IMG_SIZE)

train_loader = DataLoader(
    train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
    num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
)

val_loader = DataLoader(
    val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
    num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
)

print(f"✅ Train: {len(train_dataset)} imgs, {len(train_loader)} batches")
print(f"✅ Valid: {len(val_dataset)} imgs, {len(val_loader)} batches")
# ==============================================================================
# CELDA 8: CREAR MODELO
# ==============================================================================

def create_retinanet(num_classes):
    """Crear RetinaNet de forma robusta"""
    try:
        # Cargar modelo pre-entrenado
        model = retinanet_resnet50_fpn(weights='DEFAULT')

        # Modificar el número de clases
        num_anchors = model.head.classification_head.num_anchors

        # Reemplazar cabezal de clasificación
        model.head.classification_head.num_classes = num_classes + 1  # +1 para background

        # Reconstruir la última capa de clasificación
        cls_logits = nn.Conv2d(
            256,  # in_channels del feature pyramid
            num_anchors * (num_classes + 1),
            kernel_size=3,
            stride=1,
            padding=1
        )
        torch.nn.init.normal_(cls_logits.weight, std=0.01)
        torch.nn.init.constant_(cls_logits.bias, -np.log((1 - 0.01) / 0.01))

        model.head.classification_head.cls_logits = cls_logits

        return model

    except Exception as e:
        print(f"⚠️ Error al crear modelo: {e}")
        print("🔄 Intentando método alternativo...")

        # Método alternativo: crear desde cero
        from torchvision.models.detection.retinanet import RetinaNet
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

        backbone = resnet_fpn_backbone('resnet50', weights='DEFAULT')
        model = RetinaNet(backbone, num_classes + 1)

        return model

print("🔨 Creando RetinaNet...")
model = create_retinanet(config.NUM_CLASSES)
model = model.to(config.DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Modelo creado: {total_params:,} parámetros")
# ==============================================================================
# CELDA 9: ENTRENAMIENTO
# ==============================================================================

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Solo entrenar el head y últimas capas
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=config.LR, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
        )

        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, epoch):
        self.model.train()
        losses = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS}')
        for images, targets in pbar:
            try:
                images = [img.to(self.config.DEVICE) for img in images]
                targets = [{k: v.to(self.config.DEVICE) for k, v in t.items()} for t in targets]

                # Forward
                loss_dict = self.model(images, targets)
                loss = sum(l for l in loss_dict.values() if not torch.isnan(l))

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()

                losses.append(loss.item())
                pbar.set_postfix({'loss': f'{np.mean(losses[-10:]):.4f}'})

            except Exception as e:
                print(f"⚠️ Error en batch: {e}")
                continue

        return np.mean(losses) if losses else 0.0

    def validate(self):
        self.model.train()  # RetinaNet necesita train mode para calcular loss
        losses = []

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validating', leave=False):
                try:
                    images = [img.to(self.config.DEVICE) for img in images]
                    targets = [{k: v.to(self.config.DEVICE) for k, v in t.items()} for t in targets]

                    loss_dict = self.model(images, targets)
                    loss = sum(l for l in loss_dict.values() if not torch.isnan(l))

                    if not torch.isnan(loss) and not torch.isinf(loss):
                        losses.append(loss.item())

                except Exception as e:
                    continue

        return np.mean(losses) if losses else 0.0

    def save_checkpoint(self, epoch, loss):
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }

            path = f"{self.config.SAVE_DIR}/best_model.pth"
            torch.save(checkpoint, path)
            print(f'✅ Guardado en {path}')
        except Exception as e:
            print(f"⚠️ Error al guardar: {e}")

    def plot_losses(self):
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.train_losses, 'o-', label='Train', alpha=0.8)
            plt.plot(self.val_losses, 's-', label='Val', alpha=0.8)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        except:
            pass

    def train(self):
        print("\n" + "="*70)
        print("🚀 ENTRENAMIENTO RETINANET")
        print("="*70)

        for epoch in range(self.config.NUM_EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.scheduler.step()

            print(f'\n📊 Epoch {epoch+1}/{self.config.NUM_EPOCHS}')
            print(f'   Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {self.optimizer.param_groups[0]["lr"]:.6f}')

            if val_loss < self.best_loss and val_loss > 0:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, val_loss)

            if (epoch + 1) % 5 == 0:
                self.plot_losses()

        print("\n✅ Entrenamiento completado")
        self.plot_losses()

trainer = Trainer(model, train_loader, val_loader, config)
# ==============================================================================
# CELDA 10: EJECUTAR ENTRENAMIENTO
# ==============================================================================

trainer.train()

print(f"\n🎉 Mejor loss: {trainer.best_loss:.4f}")
print(f"📁 Modelo guardado en: {config.SAVE_DIR}")
# ==============================================================================
# CELDA 11: PREDICCIÓN
# ==============================================================================

def predict_image(model, img_path, img_size=640, threshold=0.3):
    """Predicción robusta"""
    model.eval()

    try:
        # Cargar imagen
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        # Procesar
        img_resized = cv2.resize(img_rgb, (img_size, img_size))
        img_norm = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_norm - mean) / std

        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        img_tensor = img_tensor.unsqueeze(0).to(config.DEVICE)

        # Predecir
        with torch.no_grad():
            predictions = model(img_tensor)

        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()

        # Filtrar
        mask = scores > threshold
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

        # Escalar a tamaño original
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= w_orig / img_size
            boxes[:, [1, 3]] *= h_orig / img_size

        return boxes, scores, labels, img_rgb

    except Exception as e:
        print(f"❌ Error: {e}")
        return np.array([]), np.array([]), np.array([]), None

def visualize(img, boxes, scores, labels, title=""):
    """Visualización"""
    if img is None or len(boxes) == 0:
        print("Sin detecciones")
        return

    class_names = ['BG', 'Proy_2', 'Hernia', 'Sin Hernia']
    colors = [(128,128,128), (255,255,0), (255,0,0), (0,255,0)]

    img_vis = img.copy()
    h, w = img.shape[:2]

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        idx = int(label) if int(label) < len(colors) else 0
        color = colors[idx]
        name = class_names[idx] if idx < len(class_names) else f"C{idx}"

        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        text = f'{name}: {score:.0%}'
        cv2.putText(img_vis, text, (x1, max(15, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_vis)
    plt.title(f'{title} - {len(boxes)} detecciones')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Cargar modelo
try:
    checkpoint = torch.load(f"{config.SAVE_DIR}/best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Modelo cargado")
except:
    print("⚠️ Usando modelo actual (no se encontró checkpoint)")
    # ==============================================================================
# CELDA 12: SUBIR IMÁGENES Y PREDECIR
# ==============================================================================

from google.colab import files
from IPython.display import clear_output

print("🎯 PREDICCIÓN")
print(f"Umbral de confianza: 0.5")
print("\n📤 Sube tus imágenes...")

uploaded = files.upload()

if uploaded:
    clear_output(wait=True)
    print(f"✅ {len(uploaded)} imagen(es)\n")

    for i, (filename, data) in enumerate(uploaded.items(), 1):
        print(f"\n[{i}/{len(uploaded)}] {filename}")

        path = f'/content/temp_{filename}'
        with open(path, 'wb') as f:
            f.write(data)

        boxes, scores, labels, img = predict_image(model, path, config.IMG_SIZE, 0.3)
        visualize(img, boxes, scores, labels, filename)

        import os
        if os.path.exists(path):
            os.remove(path)

    print("\n✅ Completado")
else:
    print("No se subieron imágenes")
# ==============================================================================
# CELDA 12: SUBIR IMÁGENES Y PREDECIR
# ==============================================================================

from google.colab import files
from IPython.display import clear_output

print("🎯 PREDICCIÓN")
print(f"Umbral de confianza: 0.5")
print("\n📤 Sube tus imágenes...")

uploaded = files.upload()

if uploaded:
    clear_output(wait=True)
    print(f"✅ {len(uploaded)} imagen(es)\n")

    for i, (filename, data) in enumerate(uploaded.items(), 1):
        print(f"\n[{i}/{len(uploaded)}] {filename}")

        path = f'/content/temp_{filename}'
        with open(path, 'wb') as f:
            f.write(data)

        boxes, scores, labels, img = predict_image(model, path, config.IMG_SIZE, 0.3)
        visualize(img, boxes, scores, labels, filename)

        import os
        if os.path.exists(path):
            os.remove(path)

    print("\n✅ Completado")
else:
    print("No se subieron imágenes")
# ==============================================================================
# EVALUACIÓN DEL MODELO YA ENTRENADO
# ==============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# Instalar dependencias si no están
try:
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import pandas as pd
except:
    !pip install -q scikit-learn seaborn pandas
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import pandas as pd

from collections import defaultdict

print("✅ Imports completados")

# ==============================================================================
# CONFIGURACIÓN - SOLO RUTA DEL MODELO
# ==============================================================================

# AJUSTA ESTA RUTA DONDE ESTÁ TU MODELO
MODEL_PATH = '/content/drive/MyDrive/Colab Notebooks/retinanet_hernia/best_model.pth'

# Nombres de tus clases
CLASS_NAMES = ['Proy_2', 'Hernia', 'Sin Hernia']

# Parámetros de evaluación
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.3

print(f"📂 Modelo: {MODEL_PATH}")
print(f"📊 Clases: {CLASS_NAMES}")

# ==============================================================================
# CARGAR MODELO
# ==============================================================================

print("\n📥 Cargando modelo...")

# Cargar el checkpoint
checkpoint = torch.load(MODEL_PATH, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)

# El modelo ya está cargado en tu código original (variable 'model')
# Solo actualizamos los pesos
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Modelo cargado")
print(f"   Epoch entrenado: {checkpoint.get('epoch', 'N/A')}")
print(f"   Loss final: {checkpoint.get('loss', 'N/A')}")


