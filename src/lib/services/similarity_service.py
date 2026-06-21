from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import cv2
import numpy as np

import torch
from torchvision import models, transforms
from PIL import Image
from collections import defaultdict

import os
import psycopg2
from pgvector.psycopg2 import register_vector
from uuid import uuid4
from psycopg2.extras import execute_values

from lib.schemas import EmbeddingRecord, Neighbor, SearchResult
from lib.storage.base import EmbeddingStoreProtocol

logger = logging.getLogger(__name__)


class SimilarityService:
    """Etapa 1: buscador de imagenes por similitud.

    Funciones a implementar por el estudiante:
      - extract_embedding(image)
      - search_similar_images(embedding, top_k)
      - predict_breed_from_neighbors(results)

    La orquestacion (search, index_image, persistencia y metricas de similitud)
    ya esta provista y no debe modificarse sin justificarlo en el informe.
    """

    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        top_k: int,
        image_size: int,
        model_name: str,
        url_resolver: Optional[Callable[[Path], Optional[str]]] = None,
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.image_size = image_size
        self.model_name = model_name
        self.url_resolver = url_resolver

        self.preprocess = transforms.Compose([(transforms.Resize((256),transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV)
        return image

    # ------------------------------------------------------------------
    # Etapa 1: funciones a implementar
    # ------------------------------------------------------------------

    def extract_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando un modelo pre-entrenado en
        ImageNet (ej: ResNet50, EfficientNet, ConvNeXt) sin la capa de
        clasificacion final.

        Sugerencias:
          - Preprocesar la imagen (resize a self.image_size, normalizacion ImageNet).
          - Usar torchvision.models o timm con pesos pre-entrenados.
          - Recordar que la imagen llega en BGR (OpenCV).
        Retorna una lista de floats de dimension EMBEDDING_DIM.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_img = Image.fromarray(image_rgb)
        
        tensor_img = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor_img)
            
            embedding_tensor = torch.flatten(output).cpu()
            
            embedding_list = embedding_tensor.numpy().tolist()
            
        return embedding_list

    
    def populate_vector_db(self, image_records: list[dict], batch_size: int = 32):
    
        """Puebla la base de datos de pgvector, con batching de 32: id_imagen, embedding, path, breed y metadata"""

        conn = psycopg2.connect(host=getattr(self, "db_host", "localhost"), 
            database=getattr(self, "db_name", "tu_base_datos"),
            user=getattr(self, "db_user", "postgres"),
            password=getattr(self, "db_password", "password"),
            port=getattr(self, "db_port", 5432))

        cursor = conn.cursor()
        register_vector(conn)

        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("""CREATE TABLE IF NOT EXISTS dog_embeddings(
                id_imagen TEXT PRIMARY KEY, -- str (UUID)
                path TEXT NOT NULL UNIQUE, -- str
                breed TEXT NOT NULL, -- str
                metadata JSONB NOT NULL, -- dict
                embedding vector(2048) -- list[float]);""")
        conn.commit()
        
        print(f"carga de {len(image_records)} registros vectoriales")

        for i in range(0, len(image_records), batch_size):
            sub_list = image_records[i : i + batch_size]
            
            valid_tensors = []
            valid_metadata = []

            for record in sub_list:
                path = record['path']
                breed = record['breed']
                meta_dict = record.get('metadata', {})

                if not os.path.exists(path):
                    continue

                image = cv2.imread(path)
                if image is None:
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(image_rgb)
                
                tensor_img = self.preprocess(pil_img)
                
                valid_tensors.append(tensor_img)
                valid_metadata.append((path, breed, meta_dict))

            if not valid_tensors:
                continue

            try:
                batch_tensor = torch.stack(valid_tensors).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    if len(outputs.shape) > 2:
                        outputs = torch.flatten(outputs, start_dim=1)
                    
                    embeddings_numpy = outputs.cpu().numpy()


                insert_data = []
                for idx, (path, breed, meta_dict) in enumerate(valid_metadata):
                    emb_list = embeddings_numpy[idx].tolist()
                    
                    id_imagen = str(uuid4())

                    meta_json = json.dumps(meta_dict)

                    insert_data.append((id_imagen, path, breed, meta_json, emb_list))

                query = """INSERT INTO dog_embeddings (id_imagen, path, breed, metadata, embedding) 
                    VALUES %s
                    ON CONFLICT (path) 
                    DO UPDATE SET embedding = EXCLUDED.embedding,breed = EXCLUDED.breed,metadata = EXCLUDED.metadata;"""
                execute_values(cursor, query, insert_data)
                conn.commit()

                print(f"Lote procesado {min(i + batch_size, len(image_records))}/{len(image_records)}")

            except Exception as e:
                print(f"Error al procesar el lote, índice {i}: {str(e)}")
                conn.rollback()
                continue

            cursor.close()
            conn.close()
            print("Base de datos generada")

    def search_similar_images(self, embedding: list[float], top_k: int) -> list[Neighbor]:
        """
        Recupera de la base vectorial las top_k imagenes mas similares.

        Sugerencias:
          - Con pgvector: self.store.search(embedding, top_k).
          - Con JSON: iterar self.store.all() y usar self.similarity(...).
          - Respetar SIMILARITY_METRIC (cosine | l2).
        Retorna una lista de Neighbor (path, breed, score) ordenada por score
        descendente.
        """
        raw_results = self.store.search(embedding, top_k)
        neighbors = []
        
        for item in raw_results:
            path = getattr(item, 'path', item.get('path') if isinstance(item, dict) else item[1])
            breed = getattr(item, 'breed', item.get('breed') if isinstance(item, dict) else item[2])
            ref_embedding = getattr(item, 'embedding', item.get('embedding') if isinstance(item, dict) else item[4])

            if hasattr(ref_embedding, 'tolist'):
                ref_embedding = ref_embedding.tolist()

            score = self.similarity(embedding, ref_embedding)   
            neighbor = Neighbor(path=path,breed=breed,score=score)
            neighbors.append(neighbor)
            
        neighbors.sort(key=lambda x: x.score, reverse=True)
        
        return neighbors


    def predict_breed_from_neighbors(self, results: list[Neighbor]) -> tuple[str, float]:
        """
        Predice la raza a partir de los vecinos recuperados (ej: voto
        mayoritario, opcionalmente ponderado por score).

        Si el mejor score esta por debajo de self.similarity_threshold se
        considera "unknown". Retorna (raza, score).
        """

        if not results:
            return "unknown", 0.0

        best_score = results[0].score

        threshold = getattr(self, "similarity_threshold", 0.0)
        
        if best_score < threshold:
            return "unknown", float(best_score)

        breed_votes = defaultdict(float)
        
        for neighbor in results:
            breed_votes[neighbor.breed] += neighbor.score

        predicted_breed = max(breed_votes, key=breed_votes.get)

        return predicted_breed, float(best_score)

    # ------------------------------------------------------------------
    # Helpers de similitud provistos
    # ------------------------------------------------------------------

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def index_image(
        self, image_path: str, breed: str, metadata: dict[str, object] | None = None
    ) -> EmbeddingRecord:
        """Extrae el embedding de una imagen del dataset y lo persiste en la base vectorial."""
        image = self._load_image(image_path)
        embedding = self.extract_embedding(image)
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=str(image_path),
            breed=breed,
            metadata=metadata or {},
        )
        self.store.append(record)
        return record

    def _with_url(self, neighbor: Neighbor) -> Neighbor:
        if self.url_resolver is not None and not neighbor.url:
            neighbor.url = self.url_resolver(Path(neighbor.path))
        return neighbor

    def search(
        self,
        source_path: str,
        output_path: Path,
        embedding_fn: Optional[Callable[[np.ndarray], list[float]]] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Pipeline completo de la Etapa 1: embedding -> vecinos -> raza predicha.

        `embedding_fn` permite seleccionar dinamicamente el extractor
        (baseline, resnet18_finetuned o cnn_custom, ver Etapa 2).
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        extractor = embedding_fn or self.extract_embedding
        embedding = extractor(image)

        k = int(top_k) if top_k else self.top_k
        neighbors = [self._with_url(n) for n in self.search_similar_images(embedding, k)]
        breed, score = self.predict_breed_from_neighbors(neighbors)
        logger.info("Predicted breed: %s (score=%.4f) for %s", breed, score, source_path)

        payload = SearchResult(
            source_path=source_path,
            model=model_name or self.model_name,
            predicted_breed=breed,
            score=round(float(score), 4),
            neighbors=neighbors,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
