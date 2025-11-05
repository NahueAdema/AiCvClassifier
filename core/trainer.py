import os
import json
import logging
import numpy as np
from config.settings import Settings
from models.classifier import JobMatcher

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, classifier, pipeline):
        self.classifier = classifier
        self.pipeline = pipeline

    def train_from_json(self, dataset_path: str = None) -> bool:
        logger.info("Iniciando entrenamiento del modelo con dataset real...")

        if dataset_path is None:
            dataset_path = os.path.join(Settings.DATA_DIR, 'cv_dataset.json')

        if not os.path.exists(dataset_path):
            logger.error(f"Dataset no encontrado: {dataset_path}")
            return False

        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            if not isinstance(raw_data, list):
                logger.error("El dataset debe ser una lista de objetos")
                return False

            sample_texts = []
            sample_infos = []
            sample_label_names = []

            for item in raw_data:
                cv_text = item.get('cv_text', '')
                if not cv_text:
                    parts = []
                    if item.get('titulo'):
                        parts.append(item['titulo'])
                    if item.get('resumen'):
                        parts.append(item['resumen'])
                    for exp in item.get('experiencia_laboral', []):
                        parts.append(f"{exp.get('cargo', '')} en {exp.get('empresa', '')}: {exp.get('descripcion', '')}")
                    for edu in item.get('educacion', []):
                        parts.append(f"{edu.get('titulo', '')} en {edu.get('institucion', '')}")
                    if item.get('certificaciones'):
                        parts.append("Certificaciones: " + ", ".join(item['certificaciones']))
                    cv_text = "\n".join(parts)

                all_skills = set()
                all_skills.update([s.lower() for s in item.get('habilidades', [])])
                all_skills.update([s.lower() for s in item.get('lenguajes_programacion', [])])
                all_skills.update([s.lower() for s in item.get('certificaciones', [])])

                total_exp = item.get('experiencia_años')
                if total_exp is None:
                    total_exp = sum(exp.get('años', 0) for exp in item.get('experiencia_laboral', []))

                education_level = 'other'
                if item.get('educacion'):
                    first_edu = item['educacion'][0].get('titulo', '').lower()
                    if any(kw in first_edu for kw in ['ingeniería', 'licenciatura', 'bachelor']):
                        education_level = 'bachelor'
                    elif any(kw in first_edu for kw in ['maestría', 'master']):
                        education_level = 'master'
                    elif any(kw in first_edu for kw in ['doctorado', 'phd']):
                        education_level = 'doctorate'
                    elif any(kw in first_edu for kw in ['técnico', 'technical']):
                        education_level = 'technical'

                label = item.get('label', 'Revisar')
                sample_texts.append(cv_text)
                sample_infos.append({
                    'skills': list(all_skills),
                    'experience_years': total_exp,
                    'education': education_level,
                    'email': '',
                    'phone': ''
                })
                sample_label_names.append(label)

            label_map = {"No apto": 0, "Revisar": 1, "Apto": 2}
            try:
                sample_labels = [label_map[label] for label in sample_label_names]
            except KeyError as e:
                logger.error(f"Etiqueta inválida en dataset: {e}")
                return False

            logger.info(f"Dataset cargado: {len(sample_texts)} ejemplos")

            text_sequences, feature_matrix, encoded_labels = self.pipeline.process_cv_data(
                sample_texts, sample_infos, sample_label_names
            )

            self.classifier.build_model(
                max_length=Settings.MAX_SEQUENCE_LENGTH,
                num_features=feature_matrix.shape[1]
            )

            history = self.classifier.train(
                text_sequences, feature_matrix, np.array(sample_labels)
            )

            model_path = self.classifier.save_model()
            logger.info(f"Modelo entrenado y guardado en: {model_path}")

            return True

        except Exception as e:
            logger.error(f"Error durante el entrenamiento con dataset real: {e}")
            return False