from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import onnxruntime

logger = logging.getLogger(__name__)


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self) -> None:
        """
        Entrena el clasificador de razas sobre el dataset (self.dataset_path).

        Modelo A (obligatorio): fine-tuning de ResNet18 pre-entrenado.
        Modelo B (opcional, recomendado): CNN propia.

        Debe:
          - Usar los splits train/valid definidos en la notebook.
          - Aplicar el preprocesamiento y data augmentation justificados.
          - Guardar el checkpoint resultante en self.active_checkpoint
            (ej: models/resnet18_finetuned.pth).
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import datasets, transforms, models
        from torch.utils.data import DataLoader
        import os   
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transformaciones en train y val
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dir = os.path.join(self.dataset_path, 'train')
        val_dir = os.path.join(self.dataset_path, 'valid')
        
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

        self.class_names = train_dataset.classes
        num_classes = len(self.class_names)

        # instanciar modelo
        if self.active_model_name == "resnet18_finetuned":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_ftrs = model.fc.in_features             #512 caracteristicas de la penultima capa
            model.fc = nn.Linear(num_ftrs, num_classes)     #remplazar la capa final 


        elif self.active_model_name == "cnn_custom":

            def _conv_block(in_ch: int, out_ch: int):
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )

            embedding_dim = 512
            model = nn.Sequential(
                _conv_block(3, 32),
                _conv_block(32, 64),
                _conv_block(64, 128),
                _conv_block(128, 256),
                _conv_block(256, 256),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(embedding_dim, num_classes),  # capa de clasificacio
            )



        else:
            raise ValueError(f"Modelo no soportado: {self.active_model_name}")

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 10

        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        #loop entrenamiento validacion
        for epoch in range(num_epochs):
            #fase entrenamiento
            model.train()
            running_loss = 0.0
            corrects = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = corrects.double() / len(train_dataset)

            # fase validacion
            model.eval()
            val_running_loss = 0.0
            val_corrects = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    val_running_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)
                    
            val_epoch_loss = val_running_loss / len(val_dataset)
            val_epoch_acc = val_corrects.double() / len(val_dataset)

            print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
            
            # Guardamos todo en el history
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc.item())
            self.history['val_loss'].append(val_epoch_loss)
            self.history['val_acc'].append(val_epoch_acc.item())

        os.makedirs(self.active_checkpoint.parent, exist_ok=True)
        torch.save(model, self.active_checkpoint)
        print(f"Checkpoint guardado en: {self.active_checkpoint}")







    def evaluate_classifier(self) -> dict[str, float]:
        """
        Evalua el modelo activo sobre el conjunto de prueba.

        Debe reportar: accuracy, precision, recall (sensibilidad),
        specificity (especificidad) y F1-Score. La matriz de confusion y las
        curvas de entrenamiento se documentan en la notebook.

        Retorna un dict con las metricas, ej:
          {"accuracy": 0.91, "precision": 0.90, "recall": 0.89,
           "specificity": 0.99, "f1": 0.90}
        """
        import torch
        import numpy as np
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        import os

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #cargar datos test
        test_dir = os.path.join(self.dataset_path, 'test')
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = self.load_model()
        if isinstance(model, torch.nn.Module):
            model = model.to(device)
            model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        cm = confusion_matrix(all_labels, all_preds)
        
        fp = cm.sum(axis=0) - np.diag(cm)  
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)
        specificity = np.mean(tn / (tn + fp + 1e-9))

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "specificity": float(specificity),
            "f1": float(f1),
            "confusion_matrix": cm.tolist() 
        }
    

