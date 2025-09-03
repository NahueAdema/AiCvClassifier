import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Dict
import os
import json
from config.settings import settings

class CVClassifier:
    """Modelo de clasificación de CVs usando redes neuronales"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.model = None
        self.history = None
        
    def build_model(self, max_length: int = 512, num_features: int = 10) -> Model:
        """Construye el modelo de clasificación híbrido (texto + características)"""
        
        # Input para secuencias de texto
        text_input = layers.Input(shape=(max_length,), name='text_input')
        
        # Embedding layer
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=settings.EMBEDDING_DIM,
            input_length=max_length,
            mask_zero=True
        )(text_input)
        
        # Capas LSTM bidireccionales
        lstm_out = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
        )(embedding)
        
        # Attention mechanism simplificado
        attention = layers.GlobalAveragePooling1D()(lstm_out)
        
        # Input para características numéricas
        feature_input = layers.Input(shape=(num_features,), name='feature_input')
        feature_dense = layers.Dense(32, activation='relu')(feature_input)
        feature_dropout = layers.Dropout(0.3)(feature_dense)
        
        # Concatenar características textuales y numéricas
        combined = layers.concatenate([attention, feature_dropout])
        
        # Capas de clasificación
        dense1 = layers.Dense(128, activation='relu')(combined)
        dropout1 = layers.Dropout(0.5)(dense1)
        
        dense2 = layers.Dense(64, activation='relu')(dropout1)
        dropout2 = layers.Dropout(0.3)(dense2)
        
        # Capa de salida
        output = layers.Dense(settings.NUM_CLASSES, activation='softmax', name='classification')(dropout2)
        
        # Crear modelo
        self.model = Model(inputs=[text_input, feature_input], outputs=output)
        
        # Compilar con sparse_categorical_crossentropy y accuracy
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=settings.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, text_data: np.ndarray, feature_data: np.ndarray, 
              labels: np.ndarray, validation_split: float = 0.2) -> Dict:
        """Entrena el modelo"""
        if self.model is None:
            raise ValueError("Modelo no construido. Llama a build_model() primero.")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
            )
        ]
        
        # Entrenar
        self.history = self.model.fit(
            [text_data, feature_data],
            labels,
            batch_size=settings.BATCH_SIZE,
            epochs=settings.EPOCHS,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def predict(self, text_data: np.ndarray, feature_data: np.ndarray) -> np.ndarray:
        """Realiza predicciones"""
        if self.model is None:
            raise ValueError("Modelo no entrenado.")
        
        predictions = self.model.predict([text_data, feature_data])
        return predictions
    
    def predict_single(self, text_sequence: np.ndarray, features: np.ndarray) -> Dict:
        """Predice un solo CV y devuelve resultado detallado"""
        prediction = self.predict(text_sequence.reshape(1, -1), 
                                features.reshape(1, -1))[0]
        
        classes = ['No Apto', 'Revisar', 'Apto']
        class_probabilities = {
            classes[i]: float(prediction[i]) for i in range(len(classes))
        }
        
        predicted_class = classes[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': class_probabilities
        }
    
    def calculate_cv_score(self, text_sequence: np.ndarray, features: np.ndarray,
                          job_requirements: Dict = None) -> float:
        """Calcula un score ponderado del CV (0-100)"""
        prediction = self.predict(text_sequence.reshape(1, -1), 
                                features.reshape(1, -1))[0]
        
        # Score base del modelo
        base_score = prediction[2] * 70 + prediction[1] * 30  # Apto * 70% + Revisar * 30%
        
        # Bonus por características específicas
        bonus = 0
        feature_dict = {
            'experience_years': features[4] if len(features) > 4 else 0,
            'num_tech_skills': features[0] if len(features) > 0 else 0,
            'education_level': features[5] if len(features) > 5 else 0
        }
        
        if feature_dict['experience_years'] > 0:
            bonus += min(feature_dict['experience_years'] * 2, 15)
        
        if feature_dict['num_tech_skills'] > 0:
            bonus += min(feature_dict['num_tech_skills'] * 3, 15)
        
        total_score = min(base_score * 100 + bonus, 100)
        return round(total_score, 2)
    
    def save_model(self, model_path: str = None) -> str:
        """Guarda el modelo entrenado"""
        if model_path is None:
            model_path = os.path.join(settings.MODEL_DIR, 'cv_classifier_model')
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        
        metadata = {
            'vocab_size': self.vocab_size,
            'max_length': settings.MAX_SEQUENCE_LENGTH,
            'embedding_dim': settings.EMBEDDING_DIM,
            'num_classes': settings.NUM_CLASSES
        }
        
        with open(f"{model_path}_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        return model_path
    
    def load_model(self, model_path: str) -> bool:
        """Carga un modelo previamente entrenado"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            
            metadata_path = f"{model_path}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.vocab_size = metadata.get('vocab_size', self.vocab_size)
            
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
    
    def get_model_summary(self) -> str:
        if self.model is None:
            return "Modelo no construido"
        return self.model.summary()

class JobMatcher:
    """Clase para comparar CVs con descripciones de trabajo específicas"""
    
    def __init__(self, classifier: CVClassifier):
        self.classifier = classifier
    
    def create_job_profile(self, job_description: str, required_skills: list,
                          min_experience: int = 0, preferred_education: str = 'bachelor') -> Dict:
        return {
            'description': job_description,
            'required_skills': required_skills,
            'min_experience': min_experience,
            'preferred_education': preferred_education,
            'weights': {
                'skills': 0.4,
                'experience': 0.3,
                'education': 0.2,
                'overall_quality': 0.1
            }
        }
    
    def match_cv_to_job(self, cv_text: str, cv_info: Dict, job_profile: Dict) -> Dict:
        from data.preprocessor import TextPreprocessor, FeatureExtractor
        
        text_processor = TextPreprocessor()
        feature_extractor = FeatureExtractor()
        
        processed_text = text_processor.preprocess_text(cv_text)
        text_sequence = text_processor.texts_to_sequences([processed_text])[0]
        
        features_dict = feature_extractor.extract_features(cv_info)
        features = np.array(list(features_dict.values()))
        
        base_result = self.classifier.predict_single(text_sequence, features)
        job_score = self._calculate_job_compatibility(cv_info, job_profile)
        
        final_score = (base_result['confidence'] * 0.6 + job_score * 0.4) * 100
        
        return {
            'base_classification': base_result,
            'job_compatibility_score': job_score * 100,
            'final_score': round(final_score, 2),
            'recommendation': self._get_recommendation(final_score),
            'missing_skills': self._find_missing_skills(cv_info['skills'], job_profile['required_skills'])
        }
    
    def _calculate_job_compatibility(self, cv_info: Dict, job_profile: Dict) -> float:
        score = 0.0
        weights = job_profile['weights']
        
        required_skills = set(s.lower() for s in job_profile['required_skills'])
        cv_skills = set(s.lower() for s in cv_info.get('skills', []))
        
        if required_skills:
            skills_match = len(required_skills.intersection(cv_skills)) / len(required_skills)
            score += skills_match * weights['skills']
        
        cv_experience = cv_info.get('experience_years', 0)
        min_experience = job_profile['min_experience']
        if cv_experience >= min_experience:
            experience_score = min(cv_experience / max(min_experience * 2, 5), 1.0)
        else:
            experience_score = cv_experience / max(min_experience, 1)
        score += experience_score * weights['experience']
        
        education_levels = {'doctorate': 4, 'master': 3, 'bachelor': 2, 'technical': 1, 'other': 0}
        cv_education = education_levels.get(cv_info.get('education', 'other'), 0)
        preferred_education = education_levels.get(job_profile['preferred_education'], 2)
        education_score = min(cv_education / preferred_education, 1.0) if preferred_education > 0 else 0.5
        score += education_score * weights['education']
        
        completeness = 0
        if cv_info.get('email'): completeness += 0.3
        if cv_info.get('phone'): completeness += 0.3
        if cv_info.get('skills'): completeness += 0.4
        
        score += completeness * weights['overall_quality']
        
        return min(score, 1.0)
    
    def _get_recommendation(self, score: float) -> str:
        if score >= 80:
            return "Altamente recomendado - Programar entrevista"
        elif score >= 65:
            return "Recomendado - Revisar con detalle"
        elif score >= 45:
            return "Posible candidato - Evaluación adicional"
        else:
            return "No recomendado para esta posición"
    
    def _find_missing_skills(self, cv_skills: list, required_skills: list) -> list:
        cv_skills_lower = [s.lower() for s in cv_skills]
        missing = [skill for skill in required_skills if skill.lower() not in cv_skills_lower]
        return missing
