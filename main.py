import os
import sys
import logging
from typing import List, Dict
import argparse
import tempfile
import json
from nltk_setup import download_nltk_resources

download_nltk_resources()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar módulos locales
from config.settings import Settings
from data.extractor import DocumentExtractor, CVInfoExtractor
from data.preprocessor import DataPipeline
from models.classifier import CVClassifier, JobMatcher
from config.responses import CVResponseGenerator


class CVClassifierApp:
    
    def __init__(self):
        self.document_extractor = DocumentExtractor()
        self.cv_info_extractor = CVInfoExtractor()
        self.data_pipeline = DataPipeline()
        self.classifier = CVClassifier()
        self.job_matcher = None
        
    def process_single_cv(self, file_path: str, job_requirements: Dict = None) -> Dict:
        """Procesa un CV individual y devuelve análisis completo MEJORADO"""
        try:
            logger.info(f"Procesando CV: {file_path}")
            
            # Extraer texto
            text = self.document_extractor.extract_text(file_path)
            if not text:
                return {"error": "No se pudo extraer texto del archivo"}
            
            # **VALIDACIÓN PREVIA** - Filtrar CVs no técnicos
            logger.info("Realizando validación previa...")
            initial_validation = Settings.validate_profile(text)
            
            if initial_validation == "No apto":
                logger.info("CV rechazado en validación previa - perfil no técnico")
                
                # Obtener análisis detallado para debugging
                debug_analysis = Settings.debug_profile_analysis(text)
                
                # Crear resultado básico para respuesta generada
                basic_result = {
                    'predicted_class': 'No apto',
                    'confidence': 0.95,
                    'cv_score': 0.0,
                    'tech_score': Settings.calculate_tech_score(text),
                    'cv_info': {'skills': [], 'experience_years': 0},
                    'rejection_reason': 'Perfil no técnico detectado en validación previa',
                    'debug_info': debug_analysis,
                    'text_preview': text[:200] + "..." if len(text) > 200 else text,
                    'file_path': file_path,
                    'validation_stage': 'pre_validation'
                }
                
                # Generar respuesta detallada
                detailed_response = CVResponseGenerator.generate_detailed_response(basic_result)
                basic_result['detailed_analysis'] = detailed_response
                basic_result['analysis'] = detailed_response
                
                return basic_result
            
            logger.info(f"CV pasó validación previa: {initial_validation}")
            
            # Extraer información estructurada
            cv_info = self.cv_info_extractor.extract_info(text)
            
            # **VALIDACIÓN ADICIONAL CON INFORMACIÓN EXTRAÍDA**
            tech_score = Settings.calculate_tech_score(text)
            logger.info(f"Score técnico calculado: {tech_score:.2f}")
            
            # Si el score técnico es muy bajo, rechazar
            if tech_score < 0.3:
                logger.info("CV rechazado por score técnico insuficiente")
                
                basic_result = {
                    'predicted_class': 'No apto',
                    'confidence': 0.85,
                    'cv_score': tech_score * 100,
                    'tech_score': tech_score,
                    'rejection_reason': f'Score técnico insuficiente: {tech_score:.2f}',
                    'cv_info': cv_info,
                    'text_preview': text[:200] + "..." if len(text) > 200 else text,
                    'file_path': file_path,
                    'validation_stage': 'tech_score_validation'
                }
                
                # Generar respuesta detallada
                detailed_response = CVResponseGenerator.generate_detailed_response(basic_result)
                basic_result['detailed_analysis'] = detailed_response
                basic_result['analysis'] = detailed_response
                
                return basic_result
            
            # Procesar para el modelo
            processed_text = self.data_pipeline.text_processor.preprocess_text(text)
            
            # Si no hay tokenizer, crear uno básico
            if not self.data_pipeline.text_processor.tokenizer:
                self.data_pipeline.text_processor.create_tokenizer([processed_text])
            
            text_sequence = self.data_pipeline.text_processor.texts_to_sequences([processed_text])[0]
            features_dict = self.data_pipeline.feature_extractor.extract_features(cv_info)
            features = list(features_dict.values())
            
            # Verificar si hay modelo entrenado
            model_path = os.path.join(Settings.MODEL_DIR, 'cv_classifier_model.keras')
            model_exists = os.path.exists(model_path) or os.path.exists(model_path.replace('.keras', '.h5'))
            
            if model_exists:
                # Intentar cargar el modelo
                if not self.classifier.model:
                    success = self.classifier.load_model(model_path)
                    if not success:
                        logger.warning("No se pudo cargar el modelo existente")
                
                if self.classifier.model:
                    # Realizar predicción con el modelo
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
                    # Fallback a análisis basado en reglas si el modelo no se carga
                    result = self._fallback_analysis(tech_score, cv_info)
            else:
                logger.warning("No hay modelo entrenado. Usando análisis basado en reglas.")
                result = self._fallback_analysis(tech_score, cv_info)
            
            # Añadir información extraída y de validación
            result.update({
                'cv_info': cv_info,
                'text_preview': text[:200] + "..." if len(text) > 200 else text,
                'file_path': file_path,
                'validation_stage': 'full_processing',
                'initial_validation': initial_validation,
                'tech_score': tech_score,
                'cv_score': result.get('cv_score', tech_score * 100)
            })
            
            # **GENERAR RESPUESTA DETALLADA MEJORADA**
            detailed_response = CVResponseGenerator.generate_detailed_response(result)
            result['detailed_analysis'] = detailed_response
            result['analysis'] = detailed_response  # Para compatibilidad
            
            logger.info(f"CV procesado: {result['predicted_class']} (Score: {result.get('cv_score', 0):.1f})")
            return result
            
        except Exception as e:
            logger.error(f"Error procesando CV {file_path}: {e}")
            return {"error": str(e)}
    
    def _fallback_analysis(self, tech_score: float, cv_info: Dict) -> Dict:
        """Análisis basado en reglas cuando no hay modelo disponible"""
        if tech_score >= 0.7:
            predicted_class = 'Apto'
            confidence = 0.8
        elif tech_score >= 0.4:
            predicted_class = 'Revisar' 
            confidence = 0.6
        else:
            predicted_class = 'No apto'
            confidence = 0.7
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'cv_score': tech_score * 100,
            'note': 'Modelo no entrenado - Análisis basado en reglas técnicas',
            'tech_score': tech_score
        }
    
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
        """Validación rápida de un CV MEJORADA"""
        validation_result = Settings.validate_profile(text)
        debug_analysis = Settings.debug_profile_analysis(text)
        tech_score = Settings.calculate_tech_score(text)
        
        # Crear resultado básico para la respuesta generada
        basic_result = {
            'predicted_class': validation_result,
            'confidence': 0.8 if validation_result == "Apto" else 0.6,
            'tech_score': tech_score,
            'cv_info': {'skills': [], 'experience_years': 0}
        }
        
        detailed_response = CVResponseGenerator.generate_detailed_response(basic_result)
        
        return {
            'validation_result': validation_result,
            'tech_score': tech_score,
            'debug_analysis': debug_analysis,
            'recommendation': self._get_recommendation(validation_result, tech_score),
            'detailed_assessment': detailed_response,
            'quick_analysis': detailed_response  # Para compatibilidad
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

    def train_model(self, dataset_path: str = None):
        """Entrena el modelo con un dataset real desde archivo JSON"""
        logger.info("Iniciando entrenamiento del modelo con dataset real...")
        
        # Ruta por defecto
        if dataset_path is None:
            dataset_path = os.path.join(Settings.DATA_DIR, 'cv_dataset.json')
        
        # Verificar que el archivo exista
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset no encontrado: {dataset_path}")
            return False
        
        try:
            # Cargar dataset
            with open(dataset_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Validar estructura
            if not isinstance(raw_data, list):
                logger.error("El dataset debe ser una lista de objetos")
                return False
            
            # Convertir a formato interno
            sample_texts = []
            sample_infos = []
            sample_label_names = []
            
            for item in raw_data:
                # Crear texto completo del CV
                cv_text = item.get('cv_text', '')
                if not cv_text:
                    # Si no tiene cv_text, construirlo desde otros campos
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
                
                # Unificar habilidades
                all_skills = set()
                all_skills.update([s.lower() for s in item.get('habilidades', [])])
                all_skills.update([s.lower() for s in item.get('lenguajes_programacion', [])])
                all_skills.update([s.lower() for s in item.get('certificaciones', [])])
                
                # Experiencia total
                total_exp = item.get('experiencia_años')
                if total_exp is None:
                    total_exp = sum(exp.get('años', 0) for exp in item.get('experiencia_laboral', []))
                
                # Educación (mapeo simple)
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
                
                # Etiqueta
                label = item.get('label', 'Revisar')  # Usa la etiqueta del dataset
                
                # Añadir al conjunto de entrenamiento
                sample_texts.append(cv_text)
                sample_infos.append({
                    'skills': list(all_skills),
                    'experience_years': total_exp,
                    'education': education_level,
                    'email': '',  # opcional
                    'phone': ''   # opcional
                })
                sample_label_names.append(label)
            
            # Mapear labels a números
            label_map = {"No apto": 0, "Revisar": 1, "Apto": 2}
            try:
                sample_labels = [label_map[label] for label in sample_label_names]
            except KeyError as e:
                logger.error(f"Etiqueta inválida en dataset: {e}")
                return False
            
            logger.info(f"Dataset cargado: {len(sample_texts)} ejemplos")
            
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
            logger.error(f"Error durante el entrenamiento con dataset real: {e}")
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
        success = app.train_model(args.train_data)
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
            print(f"   Análisis Detallado: {result.get('detailed_assessment', {}).get('executive_summary', '')}")
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
            
            # Mostrar resumen del análisis detallado
            if 'detailed_analysis' in result:
                analysis = result['detailed_analysis']
                print(f"   📋 Resumen: {analysis.get('executive_summary', '')}")
                print(f"   💪 Fortalezas: {len(analysis.get('strengths_analysis', {}).get('technical_skills', {}).get('top_skills', []))} habilidades técnicas")
            
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