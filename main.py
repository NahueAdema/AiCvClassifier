import os
import sys
import logging
from typing import List, Dict
import argparse
  
# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar módulos locales
from config.settings import Settings
from data.extractor import DocumentExtractor, CVInfoExtractor
from data.preprocessor import DataPipeline
from models.classifier import CVClassifier, JobMatcher

class CVClassifierApp:
    
    def __init__(self):
        self.document_extractor = DocumentExtractor()
        self.cv_info_extractor = CVInfoExtractor()
        self.data_pipeline = DataPipeline()
        self.classifier = CVClassifier()
        self.job_matcher = None
        
    def process_single_cv(self, file_path: str, job_requirements: Dict = None) -> Dict:
        """Procesa un CV individual y devuelve análisis completo"""
        try:
            logger.info(f"Procesando CV: {file_path}")
            
            # Extraer texto
            text = self.document_extractor.extract_text(file_path)
            if not text:
                return {"error": "No se pudo extraer texto del archivo"}
            
            # **NUEVA VALIDACIÓN PREVIA** - Filtrar CVs no técnicos
            logger.info("Realizando validación previa...")
            initial_validation = Settings.validate_profile(text)
            
            if initial_validation == "No apto":
                logger.info("CV rechazado en validación previa - perfil no técnico")
                
                # Obtener análisis detallado para debugging
                debug_analysis = Settings.debug_profile_analysis(text)
                
                return {
                    'predicted_class': 'No apto',
                    'confidence': 0.95,  # Alta confianza en el rechazo
                    'cv_score': 0.0,
                    'rejection_reason': 'Perfil no técnico detectado en validación previa',
                    'debug_info': debug_analysis,
                    'text_preview': text[:200] + "..." if len(text) > 200 else text,
                    'file_path': file_path,
                    'validation_stage': 'pre_validation'
                }
            
            logger.info(f"CV pasó validación previa: {initial_validation}")
            
            # Extraer información estructurada
            cv_info = self.cv_info_extractor.extract_info(text)
            
            # **VALIDACIÓN ADICIONAL CON INFORMACIÓN EXTRAÍDA**
            tech_score = Settings.calculate_tech_score(text)
            logger.info(f"Score técnico calculado: {tech_score:.2f}")
            
            # Si el score técnico es muy bajo, rechazar
            if tech_score < 0.3:
                logger.info("CV rechazado por score técnico insuficiente")
                return {
                    'predicted_class': 'No apto',
                    'confidence': 0.85,
                    'cv_score': tech_score * 100,
                    'rejection_reason': f'Score técnico insuficiente: {tech_score:.2f}',
                    'cv_info': cv_info,
                    'text_preview': text[:200] + "..." if len(text) > 200 else text,
                    'file_path': file_path,
                    'validation_stage': 'tech_score_validation'
                }
            
            # Procesar para el modelo
            processed_text = self.data_pipeline.text_processor.preprocess_text(text)
            
            # Si no hay tokenizer, crear uno básico
            if not self.data_pipeline.text_processor.tokenizer:
                self.data_pipeline.text_processor.create_tokenizer([processed_text])
            
            text_sequence = self.data_pipeline.text_processor.texts_to_sequences([processed_text])[0]
            features_dict = self.data_pipeline.feature_extractor.extract_features(cv_info)
            features = list(features_dict.values())
            
            # Verificar si hay modelo entrenado
            model_path = os.path.join(Settings.MODEL_DIR, 'cv_classifier_model')
            if os.path.exists(model_path):
                self.classifier.load_model(model_path)
                
                # Realizar predicción
                import numpy as np
                result = self.classifier.predict_single(
                    np.array(text_sequence), 
                    np.array(features)
                )
                
                score = self.classifier.calculate_cv_score(
                    np.array(text_sequence), 
                    np.array(features)
                )
                
                # **AJUSTE FINAL DEL SCORE** basado en validación técnica
                # Si pasó todas las validaciones pero el modelo da score bajo, 
                # ajustar con el tech_score
                if score < 50 and tech_score > 0.6:
                    adjusted_score = max(score, tech_score * 100)
                    logger.info(f"Score ajustado de {score} a {adjusted_score} por tech_score alto")
                    score = adjusted_score
                
                # Si hay requisitos de trabajo, hacer matching específico
                if job_requirements and self.job_matcher:
                    job_match = self.job_matcher.match_cv_to_job(text, cv_info, job_requirements)
                    result.update(job_match)
                else:
                    result['cv_score'] = score
                
                # **VALIDACIÓN FINAL** - Si el modelo dice "Apto" pero score muy bajo, degradar
                if result.get('predicted_class') == 'Apto' and score < Settings.APTO_THRESHOLD * 100:
                    result['predicted_class'] = 'Revisar'
                    result['adjustment_reason'] = f'Degradado de Apto a Revisar por score bajo: {score:.1f}'
                
            else:
                logger.warning("No hay modelo entrenado. Usando análisis basado en reglas.")
                
                # **ANÁLISIS BASADO EN REGLAS** cuando no hay modelo
                if tech_score >= 0.7:
                    predicted_class = 'Apto'
                    confidence = 0.8
                elif tech_score >= 0.4:
                    predicted_class = 'Revisar' 
                    confidence = 0.6
                else:
                    predicted_class = 'No apto'
                    confidence = 0.7
                
                result = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'cv_score': tech_score * 100,
                    'note': 'Modelo no entrenado - Análisis basado en reglas técnicas',
                    'tech_score': tech_score
                }
            
            # Añadir información extraída y de validación
            result.update({
                'cv_info': cv_info,
                'text_preview': text[:200] + "..." if len(text) > 200 else text,
                'file_path': file_path,
                'validation_stage': 'full_processing',
                'initial_validation': initial_validation,
                'tech_score': tech_score
            })
            
            logger.info(f"CV procesado: {result['predicted_class']} (Score: {result.get('cv_score', 0):.1f})")
            return result
            
        except Exception as e:
            logger.error(f"Error procesando CV {file_path}: {e}")
            return {"error": str(e)}
    
    def batch_process_cvs(self, cv_folder: str, output_file: str = None) -> List[Dict]:
        """Procesa múltiples CVs en una carpeta"""
        results = []
        
        # Buscar archivos de CV
        cv_extensions = ['.pdf', '.docx', '.doc', '.txt']
        cv_files = []
        
        for file in os.listdir(cv_folder):
            if any(file.lower().endswith(ext) for ext in cv_extensions):
                cv_files.append(os.path.join(cv_folder, file))
        
        logger.info(f"Encontrados {len(cv_files)} archivos para procesar")
        
        # **ESTADÍSTICAS DE PROCESAMIENTO**
        stats = {
            'total': len(cv_files),
            'apto': 0,
            'revisar': 0,
            'no_apto': 0,
            'errores': 0,
            'rechazados_prevalidacion': 0
        }
        
        # Procesar cada archivo
        for i, cv_file in enumerate(cv_files, 1):
            logger.info(f"Procesando {i}/{len(cv_files)}: {os.path.basename(cv_file)}")
            result = self.process_single_cv(cv_file)
            results.append(result)
            
            # Actualizar estadísticas
            if 'error' in result:
                stats['errores'] += 1
            else:
                classification = result.get('predicted_class', 'Error')
                if classification == 'Apto':
                    stats['apto'] += 1
                elif classification == 'Revisar':
                    stats['revisar'] += 1
                elif classification == 'No apto':
                    stats['no_apto'] += 1
                    if result.get('validation_stage') == 'pre_validation':
                        stats['rechazados_prevalidacion'] += 1
        
        # Mostrar estadísticas
        logger.info("=" * 50)
        logger.info("ESTADÍSTICAS DE PROCESAMIENTO:")
        logger.info(f"Total procesados: {stats['total']}")
        logger.info(f"✅ Aptos: {stats['apto']}")
        logger.info(f"🔍 Revisar: {stats['revisar']}")
        logger.info(f"❌ No aptos: {stats['no_apto']}")
        logger.info(f"   - Rechazados en pre-validación: {stats['rechazados_prevalidacion']}")
        logger.info(f"⚠️  Errores: {stats['errores']}")
        logger.info("=" * 50)
        
        # Guardar resultados si se especifica
        if output_file:
            import json
            output_data = {
                'statistics': stats,
                'results': results,
                'processing_info': {
                    'total_files': len(cv_files),
                    'settings_used': {
                        'min_tech_skills': Settings.MIN_TECH_SKILLS_REQUIRED,
                        'apto_threshold': Settings.APTO_THRESHOLD,
                        'no_apto_threshold': Settings.NO_APTO_THRESHOLD
                    }
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Resultados guardados en {output_file}")
        
        return results
    
    def validate_cv_quick(self, text: str) -> Dict:
        """Validación rápida de un CV (solo pre-validación)"""
        validation_result = Settings.validate_profile(text)
        debug_analysis = Settings.debug_profile_analysis(text)
        tech_score = Settings.calculate_tech_score(text)
        
        return {
            'validation_result': validation_result,
            'tech_score': tech_score,
            'debug_analysis': debug_analysis,
            'recommendation': self._get_recommendation(validation_result, tech_score)
        }
    
    def _get_recommendation(self, validation: str, tech_score: float) -> str:
        """Genera recomendación basada en validación y score"""
        if validation == "No apto":
            return "❌ Rechazar - No cumple criterios técnicos mínimos"
        elif tech_score >= 0.7:
            return "✅ Procesar - Perfil técnico sólido"
        elif tech_score >= 0.4:
            return "🔍 Revisar - Perfil técnico medio, requiere evaluación detallada"
        else:
            return "⚠️  Cuidado - Score técnico bajo, revisar cuidadosamente"
    
    def train_model_with_sample_data(self, sample_data_path: str = None):
        """Entrena el modelo con datos de ejemplo"""
        logger.info("Iniciando entrenamiento del modelo...")
        
        # **DATOS DE EJEMPLO MEJORADOS** con casos de rechazo
        sample_texts = [
            "Ingeniero de Software con 5 años de experiencia en Python, JavaScript y React. MBA en Administración.",
            "Desarrollador Junior con conocimientos básicos en HTML, CSS. Recién graduado de carrera técnica.",
            "Senior Data Scientist con PhD en Estadística, 8 años en machine learning, TensorFlow, AWS.",
            "Cajero de supermercado con 3 años de experiencia en atención al cliente y manejo de caja registradora.",
            "Project Manager certificado PMP, 10 años liderando equipos de desarrollo, Agile, Scrum.",
            "Mesero con experiencia en restaurantes, excelente atención al cliente y trabajo en equipo.",
            "Desarrollador Full Stack Python/React con experiencia en microservicios y Docker.",
            "Guardia de seguridad con 5 años de experiencia en vigilancia y control de accesos."
        ]
        
        sample_infos = [
            {'skills': ['python', 'javascript', 'react'], 'experience_years': 5, 'education': 'master', 'email': 'test@email.com', 'phone': '123456789'},
            {'skills': ['html', 'css'], 'experience_years': 0, 'education': 'technical', 'email': 'junior@email.com', 'phone': '987654321'},
            {'skills': ['python', 'tensorflow', 'aws', 'machine learning'], 'experience_years': 8, 'education': 'doctorate', 'email': 'senior@email.com', 'phone': '555666777'},
            {'skills': ['atención al cliente'], 'experience_years': 3, 'education': 'high_school', 'email': 'cajero@email.com', 'phone': '111222333'},
            {'skills': ['liderazgo', 'trabajo en equipo', 'agile'], 'experience_years': 10, 'education': 'master', 'email': 'pm@email.com', 'phone': '444555666'},
            {'skills': ['atención al cliente', 'trabajo en equipo'], 'experience_years': 2, 'education': 'high_school', 'email': 'mesero@email.com', 'phone': '666777888'},
            {'skills': ['python', 'react', 'docker', 'microservices'], 'experience_years': 4, 'education': 'bachelor', 'email': 'fullstack@email.com', 'phone': '777888999'},
            {'skills': ['vigilancia', 'seguridad'], 'experience_years': 5, 'education': 'high_school', 'email': 'guardia@email.com', 'phone': '888999000'}
        ]
        
        # Labels corregidos: 0: No Apto, 1: Revisar, 2: Apto
        sample_labels = [2, 1, 2, 0, 2, 0, 2, 0]  # Cajero, mesero y guardia = No Apto
        sample_label_names = ['Apto', 'Revisar', 'Apto', 'No Apto', 'Apto', 'No Apto', 'Apto', 'No Apto']
        
        try:
            # Procesar datos
            text_sequences, feature_matrix, encoded_labels = self.data_pipeline.process_cv_data(
                sample_texts, sample_infos, sample_label_names
            )
            
            # Construir modelo
            self.classifier.build_model(
                max_length=Settings.MAX_SEQUENCE_LENGTH,
                num_features=feature_matrix.shape[1]
            )
            
            # Entrenar
            import numpy as np
            history = self.classifier.train(
                text_sequences, feature_matrix, np.array(sample_labels)
            )
            
            # Guardar modelo
            model_path = self.classifier.save_model()
            logger.info(f"Modelo entrenado y guardado en: {model_path}")
            
            # Crear job matcher
            self.job_matcher = JobMatcher(self.classifier)
            
            return True
            
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {e}")
            return False
    
    def create_job_profile(self, description: str, skills: List[str], 
                          min_experience: int = 0, education: str = 'bachelor') -> Dict:
        """Crea un perfil de trabajo para matching"""
        if not self.job_matcher:
            self.job_matcher = JobMatcher(self.classifier)
        
        return self.job_matcher.create_job_profile(
            description, skills, min_experience, education
        )

def main():
    """Función principal con interfaz de línea de comandos"""
    parser = argparse.ArgumentParser(description='Clasificador de CVs con IA - Con filtro de perfiles técnicos')
    parser.add_argument('--mode', choices=['train', 'process', 'batch', 'validate', 'server'], 
                       default='process', help='Modo de operación')
    parser.add_argument('--input', help='Archivo CV o carpeta para procesar')
    parser.add_argument('--output', help='Archivo de salida para resultados')
    parser.add_argument('--train-data', help='Ruta a datos de entrenamiento')
    parser.add_argument('--text', help='Texto del CV para validación rápida')
    
    args = parser.parse_args()
    
    app = CVClassifierApp()
    
    if args.mode == 'train':
        logger.info("Modo entrenamiento")
        success = app.train_model_with_sample_data(args.train_data)
        if success:
            print("✅ Modelo entrenado exitosamente")
        else:
            print("❌ Error en el entrenamiento")
            
    elif args.mode == 'validate':
        if args.text:
            result = app.validate_cv_quick(args.text)
            print(f"🔍 VALIDACIÓN RÁPIDA:")
            print(f"   Resultado: {result['validation_result']}")
            print(f"   Score técnico: {result['tech_score']:.2f}")
            print(f"   Recomendación: {result['recommendation']}")
        elif args.input:
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
            result = app.validate_cv_quick(text)
            print(f"🔍 VALIDACIÓN DE ARCHIVO: {args.input}")
            print(f"   Resultado: {result['validation_result']}")
            print(f"   Score técnico: {result['tech_score']:.2f}")
            print(f"   Recomendación: {result['recommendation']}")
        else:
            print("❌ Para validación usa --text 'texto del CV' o --input archivo.txt")
            
    elif args.mode == 'process':
        if not args.input:
            print("❌ Debes especificar un archivo con --input")
            return
        
        logger.info(f"Procesando CV: {args.input}")
        result = app.process_single_cv(args.input)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ CV procesado exitosamente:")
            print(f"   📄 Archivo: {os.path.basename(result['file_path'])}")
            print(f"   🎯 Clasificación: {result['predicted_class']}")
            print(f"   📊 Score: {result.get('cv_score', 0):.1f}/100")
            print(f"   🔧 Score técnico: {result.get('tech_score', 0):.2f}")
            print(f"   ⚡ Etapa: {result.get('validation_stage', 'unknown')}")
            
            if result.get('rejection_reason'):
                print(f"   ❌ Razón rechazo: {result['rejection_reason']}")
            
    elif args.mode == 'batch':
        if not args.input:
            print("❌ Debes especificar una carpeta con --input")
            return
        
        logger.info(f"Procesamiento en lote: {args.input}")
        results = app.batch_process_cvs(args.input, args.output)
        print(f"✅ Procesados {len(results)} CVs")
        
        # Mostrar resumen por consola
        apto = sum(1 for r in results if r.get('predicted_class') == 'Apto')
        revisar = sum(1 for r in results if r.get('predicted_class') == 'Revisar')
        no_apto = sum(1 for r in results if r.get('predicted_class') == 'No apto')
        errores = sum(1 for r in results if 'error' in r)
        
        print(f"📊 RESUMEN:")
        print(f"   ✅ Aptos: {apto}")
        print(f"   🔍 Revisar: {revisar}")
        print(f"   ❌ No aptos: {no_apto}")
        print(f"   ⚠️  Errores: {errores}")
        
    elif args.mode == 'server':
        print("🚀 Iniciando servidor API...")
        from api.endpoints import start_server
        start_server(app)

if __name__ == "__main__":
    main()