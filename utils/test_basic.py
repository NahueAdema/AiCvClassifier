import unittest
import tempfile
import os
from unittest.mock import Mock, patch
import sys

# Añadir el directorio padre al path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.extractor import CVInfoExtractor, DocumentExtractor
from data.preprocessor import TextPreprocessor, FeatureExtractor
from models.classifier import CVClassifier
from utils.text_utils import extract_years_experience, extract_programming_languages

class TestCVInfoExtractor(unittest.TestCase):
    
    def setUp(self):
        self.extractor = CVInfoExtractor()
    
    def test_extract_email(self):
        text = "Mi email es juan.perez@gmail.com y me pueden contactar ahí"
        result = self.extractor.extract_info(text)
        self.assertEqual(result['email'], 'juan.perez@gmail.com')
    
    def test_extract_skills(self):
        text = "Tengo experiencia en Python, JavaScript y React. También trabajo con Docker."
        result = self.extractor.extract_info(text)
        expected_skills = ['python', 'javascript', 'react', 'docker']
        
        found_skills = [skill.lower() for skill in result['skills']]
        for skill in expected_skills:
            self.assertIn(skill, found_skills)
    
    def test_estimate_experience(self):
        text = "Tengo 5 años de experiencia en desarrollo de software"
        result = self.extractor.extract_info(text)
        self.assertEqual(result['experience_years'], 5)
    
    def test_extract_education_level(self):
        test_cases = [
            ("Licenciatura en Ingeniería", "bachelor"),
            ("MBA en Administración", "master"),
            ("PhD en Computer Science", "doctorate"),
            ("Técnico en programación", "technical")
        ]
        
        for text, expected_level in test_cases:
            result = self.extractor.extract_info(text)
            self.assertEqual(result['education'], expected_level)

class TestTextPreprocessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = TextPreprocessor()
    
    def test_clean_text(self):
        dirty_text = "¡Hola! Mi email es test@email.com y mi teléfono +1-234-567-8900."
        clean_text = self.processor.clean_text(dirty_text)
        
        # No debe contener email ni teléfono
        self.assertNotIn('@', clean_text)
        self.assertNotIn('+', clean_text)
        self.assertNotIn('-', clean_text)
    
    def test_tokenize_and_filter(self):
        text = "Soy desarrollador con experiencia en Python y JavaScript"
        tokens = self.processor.tokenize_and_filter(text)
        
        # No debe contener stopwords como 'en', 'con'
        self.assertNotIn('en', tokens)
        self.assertNotIn('con', tokens)
        # Debe contener palabras importantes
        self.assertTrue(any('python' in token for token in tokens))
        self.assertTrue(any('javascript' in token for token in tokens))

class TestFeatureExtractor(unittest.TestCase):
    
    def setUp(self):
        self.extractor = FeatureExtractor()
    
    def test_extract_features(self):
        cv_info = {
            'skills': ['python', 'react', 'liderazgo'],
            'experience_years': 3,
            'education': 'bachelor',
            'email': 'test@email.com',
            'phone': '123456789'
        }
        
        features = self.extractor.extract_features(cv_info)
        
        self.assertEqual(features['num_tech_skills'], 2)  # python, react
        self.assertEqual(features['num_soft_skills'], 1)  # liderazgo
        self.assertEqual(features['experience_years'], 3.0)
        self.assertEqual(features['education_level'], 2.0)  # bachelor
        self.assertEqual(features['has_email'], 1.0)
        self.assertEqual(features['has_phone'], 1.0)

class TestUtilityFunctions(unittest.TestCase):
    
    def test_extract_years_experience(self):
        test_cases = [
            ("Tengo 5 años de experiencia", [5]),
            ("3 years experience in software", [3]),
            ("Experiencia de 7 años en el área", [7]),
            ("No experience mentioned", [])
        ]
        
        for text, expected in test_cases:
            result = extract_years_experience(text)
            self.assertEqual(result, expected)
    
    def test_extract_programming_languages(self):
        text = "Programo en Python, Java y JavaScript desde hace años"
        result = extract_programming_languages(text)
        
        expected_langs = {'python', 'java', 'javascript'}
        self.assertTrue(expected_langs.issubset(result))

class TestCVClassifier(unittest.TestCase):
    
    def setUp(self):
        self.classifier = CVClassifier(vocab_size=1000)
    
    def test_build_model(self):
        model = self.classifier.build_model(max_length=100, num_features=8)
        self.assertIsNotNone(model)
        
        # Verificar que el modelo tiene las entradas correctas
        input_names = [inp.name for inp in model.inputs]
        self.assertIn('text_input', input_names)
        self.assertIn('feature_input', input_names)
    
    @patch('tensorflow.keras.models.load_model')
    def test_load_model(self, mock_load):
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        # Crear archivo de metadatos temporal
        with tempfile.NamedTemporaryFile(mode='w', suffix='_metadata.json', delete=False) as f:
            f.write('{"vocab_size": 5000, "embedding_dim": 128}')
            metadata_path = f.name
        
        model_path = metadata_path.replace('_metadata.json', '')
        
        try:
            success = self.classifier.load_model(model_path)
            self.assertTrue(success)
            self.assertEqual(self.classifier.vocab_size, 5000)
        finally:
            os.unlink(metadata_path)

class TestIntegration(unittest.TestCase):
    
    def test_full_pipeline_sample(self):
        """Test de integración con datos de ejemplo"""
        sample_text = """
        Juan Pérez
        Email: juan.perez@email.com
        Teléfono: +1-555-123-4567
        
        Experiencia:
        - 3 años como Desarrollador Python
        - Especialista en Django y React
        - Trabajo en equipo y liderazgo
        
        Educación:
        - Licenciatura en Ingeniería de Sistemas
        """
        
        # Extraer información
        extractor = CVInfoExtractor()
        cv_info = extractor.extract_info(sample_text)
        
        # Verificaciones básicas
        self.assertEqual(cv_info['email'], 'juan.perez@email.com')
        self.assertIn('python', [s.lower() for s in cv_info['skills']])
        self.assertEqual(cv_info['experience_years'], 3)
        self.assertEqual(cv_info['education'], 'bachelor')
        
        # Procesar texto
        processor = TextPreprocessor()
        processed_text = processor.preprocess_text(sample_text)
        self.assertIsInstance(processed_text, str)
        self.assertGreater(len(processed_text), 0)
        
        # Extraer características
        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract_features(cv_info)
        self.assertIsInstance(features, dict)
        self.assertGreater(features['total_skills'], 0)

if __name__ == '__main__':
    # Configurar logging para tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main()