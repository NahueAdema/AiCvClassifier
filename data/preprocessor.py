import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple
import tensorflow as tf


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    
    
    def __init__(self, language='spanish'):
        self.language = language
        self.stemmer = SnowballStemmer(language)
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            self.stop_words = set()
        
        # Añadir stopwords personalizadas
        custom_stops = {'cv', 'curriculum', 'vitae', 'resume', 'experiencia', 'experience'}
        self.stop_words.update(custom_stops)
        
        self.tokenizer = None
        
    def clean_text(self, text: str) -> str:
        """Limpia el texto removiendo caracteres especiales y normalizando"""
        # Convertir a minúsculas
        text = text.lower()
        
        # Remover URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remover emails (ya extraídos por separado)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remover números de teléfono (ya extraídos por separado)
        text = re.sub(r'[\+]?[1-9]?[0-9]{7,15}', '', text)
        
        # Mantener solo letras, números y espacios
        text = re.sub(r'[^a-záéíóúñü0-9\s]', ' ', text)
        
        # Remover múltiples espacios
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_and_filter(self, text: str) -> List[str]:
        """Tokeniza y filtra stopwords"""
        tokens = word_tokenize(text, language=self.language)
        
        # Filtrar stopwords y palabras muy cortas
        filtered_tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return filtered_tokens
    
    def preprocess_text(self, text: str) -> str:
        """Pipeline completo de preprocesamiento"""
        clean_text = self.clean_text(text)
        tokens = self.tokenize_and_filter(clean_text)
        return ' '.join(tokens)
    
    def create_tokenizer(self, texts: List[str], vocab_size: int = 10000):
        """Crea y entrena un tokenizador de TensorFlow"""
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=vocab_size,
            oov_token='<OOV>',
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        self.tokenizer.fit_on_texts(texts)
        return self.tokenizer
    
    def texts_to_sequences(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """Convierte textos a secuencias numéricas paddeadas"""
        if not self.tokenizer:
            raise ValueError("Tokenizer no ha sido creado. Llama a create_tokenizer primero.")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=max_length, padding='post', truncating='post'
        )
        return padded

class FeatureExtractor:
    """Extrae características numéricas adicionales del CV"""
    
    def __init__(self):
        self.label_encoders = {}
    
    def extract_features(self, cv_info: Dict) -> Dict[str, float]:
        """Extrae características numéricas del CV"""
        features = {}
        
        # Características de habilidades
        features['num_tech_skills'] = self._count_tech_skills(cv_info.get('skills', []))
        features['num_soft_skills'] = self._count_soft_skills(cv_info.get('skills', []))
        features['total_skills'] = len(cv_info.get('skills', []))
        
        # Experiencia
        features['experience_years'] = float(cv_info.get('experience_years', 0))
        features['has_experience'] = 1.0 if features['experience_years'] > 0 else 0.0
        
        # Educación (codificada)
        education_levels = {
            'doctorate': 4.0, 'master': 3.0, 'bachelor': 2.0, 
            'technical': 1.0, 'other': 0.0
        }
        features['education_level'] = education_levels.get(
            cv_info.get('education', 'other'), 0.0
        )
        
        # Completitud del CV
        features['has_email'] = 1.0 if cv_info.get('email') else 0.0
        features['has_phone'] = 1.0 if cv_info.get('phone') else 0.0
        
        return features
    
    def _count_tech_skills(self, skills: List[str]) -> int:
        from config.settings import settings
        return sum(1 for skill in skills if skill.lower() in 
                  [s.lower() for s in settings.TECH_SKILLS])
    
    def _count_soft_skills(self, skills: List[str]) -> int:
        from config.settings import settings
        return sum(1 for skill in skills if skill.lower() in 
                  [s.lower() for s in settings.SOFT_SKILLS])
    
    def create_feature_vector(self, features_list: List[Dict]) -> np.ndarray:
        """Crea matriz de características numéricas"""
        if not features_list:
            return np.array([])
        
        # Obtener todas las claves de características
        all_keys = set()
        for features in features_list:
            all_keys.update(features.keys())
        
        # Crear matriz
        feature_matrix = []
        for features in features_list:
            row = [features.get(key, 0.0) for key in sorted(all_keys)]
            feature_matrix.append(row)
        
        return np.array(feature_matrix)

class DataPipeline:
    """Pipeline completo de procesamiento de datos"""
    
    def __init__(self):
        self.text_processor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.label_encoder = LabelEncoder()
    
    def process_cv_data(self, cv_texts: List[str], cv_infos: List[Dict], 
                       labels: List[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Procesa datos de CVs completos"""
        # Procesar textos
        processed_texts = [self.text_processor.preprocess_text(text) for text in cv_texts]
        
        # Crear tokenizer si no existe
        if not self.text_processor.tokenizer:
            self.text_processor.create_tokenizer(processed_texts)
        
        # Convertir a secuencias
        text_sequences = self.text_processor.texts_to_sequences(processed_texts)
        
        # Extraer características numéricas
        features_list = [self.feature_extractor.extract_features(info) for info in cv_infos]
        feature_matrix = self.feature_extractor.create_feature_vector(features_list)
        
        # Procesar etiquetas si se proporcionan
        encoded_labels = None
        if labels:
            encoded_labels = self.label_encoder.fit_transform(labels)
        
        return text_sequences, feature_matrix, encoded_labels