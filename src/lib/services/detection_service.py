from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from lib.schemas import ClassifyResult, DetectResult, DogDetection
from lib.services.classifier_service import ClassifierService

logger = logging.getLogger(__name__)


class DetectionService:
    """Etapa 3: pipeline de deteccion y clasificacion.

    Funciones a implementar por el estudiante:
      - detect_dogs(image)
      - classify_detected_dog(crop)

    La orquestacion (predict: deteccion -> recorte -> clasificacion -> JSON)
    ya esta provista.
    """

    def __init__(
        self,
        classifier: ClassifierService,
        yolo_model: str,
        conf_threshold: float,
        dog_class_id: int,
    ) -> None:
        self.classifier = classifier
        self.yolo_model_name = yolo_model
        self.conf_threshold = conf_threshold
        self.dog_class_id = dog_class_id

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV / ultralytics)
        return image





    # ------------------------------------------------------------------
    # Etapa 3: funciones a implementar
    # ------------------------------------------------------------------

    def detect_dogs(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """
        Detecta todos los perros presentes en la imagen usando un modelo YOLO
        pre-entrenado (ej: YOLOv8n via ultralytics). No es necesario entrenar
        el detector.

        Sugerencias:
          - self.yolo_model_name, self.conf_threshold y self.dog_class_id
            (clase 'dog' = 16 en COCO) vienen de la configuracion (.env).
          - Debe funcionar con un perro, multiples perros y escenas complejas.

        Retorna una lista de ((x1, y1, x2, y2), confidence) en pixeles.
        """
        from ultralytics import YOLO

        # Cacheamos el modelo en la instancia para no recargarlo de disco en cada llamada
        if not hasattr(self, "_yolo_model"):
            self._yolo_model = YOLO(self.yolo_model_name)
        
        # Hacemos la inferencia filtrando directamente por nuestro umbral de confianza
        results = self._yolo_model(image, conf=self.conf_threshold)
        
        detections = []
        # iteramos sobre las cajas delimitadoras de la imagen
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            
            # Filtramos estrictamente por la clase perr
            if class_id == self.dog_class_id:
                # Extraemos y convertimos coordenadas a enteros
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Extraemos confianza
                conf = float(box.conf[0].item())
                
                # Respetamos el output esperado por la orquestación: ((x1, y1, x2, y2), score)
                detections.append(((x1, y1, x2, y2), conf))
                
        return detections
    






    def classify_detected_dog(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Clasifica la raza del recorte de un perro detectado usando el modelo
        entrenado en la Etapa 2 (self.classifier.load_model()).

        El recorte llega en BGR (OpenCV). Retorna (raza, score).
        """
        import cv2
        import torch
        import torch.nn.functional as F
        from torchvision import transforms
        from PIL import Image

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #la imagen de yolo llega en bgr la convertimos a rgb y luego a PIL para poder aplicar las transformaciones
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Usamos self.classifier.image_size dinámico para las transformaciones
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.classifier.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_t = transform(pil_img).unsqueeze(0).to(device)

        # Inferencia, carga y preparación del modelo mediante el ClassifierService
        model = self.classifier.load_model()
        
        if not isinstance(model, torch.nn.Module):
             raise ValueError("error")
             
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(img_t)
            # Pasamos los logits a probabilidades (0 a 1) mediante Softmax
            probs = F.softmax(outputs, dim=1)
            # Extraemos el valor máximo de score y su posición índice predicho
            conf, pred_idx = torch.max(probs, 1)

        pred_class_idx = int(pred_idx.item())
        confidence = float(conf.item())

        # Traducción segura de Índice a Raza
        # PyTorch clasifica alfabéticamente leyendo los directorios de 'train'
        if hasattr(self.classifier, "class_names") and self.classifier.class_names:
            class_names = self.classifier.class_names
        else:
            train_dir = self.classifier.dataset_path / 'train'
            class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

        breed_name = class_names[pred_class_idx]

        # Respetamos el output esperado por classify_image y predict: (raza, score)
        return breed_name, confidence






    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def classify_image(
        self, source_path: str, output_path: Path, model_name: str | None = None
    ) -> str:
        """Clasifica la imagen completa con el modelo entrenado (pestaña Etapa 2).

        Reutiliza classify_detected_dog tratando la imagen entera como recorte,
        por lo que requiere la Etapa 2 (modelo entrenado) y classify_detected_dog.
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        if model_name:
            self.classifier.set_active_model(model_name)
        breed, score = self.classify_detected_dog(image)
        payload = ClassifyResult(
            source_path=source_path,
            model=model_name or self.classifier.active_model_name,
            breed=breed,
            score=round(float(score), 4),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)

    def predict(self, source_path: str, output_path: Path) -> str:
        """Flujo completo: deteccion -> bounding boxes -> recortes -> clasificacion.

        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        height, width = image.shape[:2]

        detections: list[DogDetection] = []
        for (box, det_score) in self.detect_dogs(image):
            x1, y1, x2, y2 = self._clip_xyxy(*[int(v) for v in box], height, width)
            crop = image[y1:y2, x1:x2]
            breed, breed_score = self.classify_detected_dog(crop)
            detections.append(
                DogDetection(
                    bbox=[x1, y1, x2, y2],
                    det_score=round(float(det_score), 4),
                    breed=breed,
                    breed_score=round(float(breed_score), 4),
                )
            )

        detected_breeds = sorted({item.breed for item in detections if item.breed != "unknown"})
        payload = DetectResult(
            source_path=source_path,
            detections=detections,
            detected_breeds=detected_breeds,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
