import os
import logging
import json
import numpy as np
from config.settings import Settings
from config.responses import CVResponseGenerator
from core.validator import CVValidator
from models.classifier import JobMatcher

logger = logging.getLogger(__name__)


class CVProcessor:
    def __init__(self, app):
        self.app = app

    def process_single_cv(self, file_path: str, job_requirements: dict = None) -> dict:
        try:
            logger.info(f"Procesando CV: {file_path}")
            text = self.app.document_extractor.extract_text(file_path)
            if not text:
                return {"error": "No se pudo extraer texto del archivo"}

            # ✅ Validación previa: perfil no técnico
            if not CVValidator.is_technical_profile(text):
                logger.info("CV rechazado en validación previa - perfil no técnico")
                return CVValidator.get_rejection_result(
                    file_path, text,
                    "Perfil no técnico detectado en validación previa",
                    "pre_validation"
                )

            logger.info("CV pasó validación previa")

            # Extraer info estructurada
            cv_info = self.app.cv_info_extractor.extract_info(text)
            tech_score = CVValidator.calculate_tech_score(text)
            logger.info(f"Score técnico calculado: {tech_score:.2f}")

            # ❌ Rechazo por score técnico bajo
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
                detailed_response = CVResponseGenerator.generate_detailed_response(basic_result)
                basic_result['detailed_analysis'] = detailed_response
                basic_result['analysis'] = detailed_response
                return basic_result

            # Preprocesar para modelo
            processed_text = self.app.data_pipeline.text_processor.preprocess_text(text)
            if not self.app.data_pipeline.text_processor.tokenizer:
                self.app.data_pipeline.text_processor.create_tokenizer([processed_text])
            text_sequence = self.app.data_pipeline.text_processor.texts_to_sequences([processed_text])[0]
            features_dict = self.app.data_pipeline.feature_extractor.extract_features(cv_info)
            features = list(features_dict.values())

            # Verificar modelo
            model_path = os.path.join(Settings.MODEL_DIR, 'cv_classifier_model.keras')
            model_exists = os.path.exists(model_path) or os.path.exists(model_path.replace('.keras', '.h5'))

            if model_exists and not self.app.classifier.model:
                success = self.app.classifier.load_model(model_path)
                if not success:
                    logger.warning("No se pudo cargar el modelo existente")

            if self.app.classifier.model:
                result = self.app.classifier.predict_single(np.array([text_sequence]), np.array([features]))
                score = self.app.classifier.calculate_cv_score(np.array([text_sequence]), np.array([features]))

                # Ajuste por tech_score alto
                if score < 50 and tech_score > 0.6:
                    adjusted_score = max(score, tech_score * 100)
                    logger.info(f"Score ajustado de {score} a {adjusted_score} por tech_score alto")
                    score = adjusted_score

                result['cv_score'] = score

                # Matching con trabajo
                if job_requirements:
                    if not self.app.job_matcher:
                        self.app.job_matcher = JobMatcher(self.app.classifier)
                    job_match = self.app.job_matcher.match_cv_to_job(text, cv_info, job_requirements)
                    result.update(job_match)

                # Degradar si score bajo pero clasificado como "Apto"
                if result.get('predicted_class') == 'Apto' and score < Settings.APTO_THRESHOLD * 100:
                    result['predicted_class'] = 'Revisar'
                    result['adjustment_reason'] = f'Degradado de Apto a Revisar por score bajo: {score:.1f}'

            else:
                logger.warning("No hay modelo entrenado. Usando análisis basado en reglas.")
                result = self._fallback_analysis(tech_score, cv_info)

            # Completar resultado
            result.update({
                'cv_info': cv_info,
                'text_preview': text[:200] + "..." if len(text) > 200 else text,
                'file_path': file_path,
                'validation_stage': 'full_processing',
                'initial_validation': Settings.validate_profile(text),
                'tech_score': tech_score,
                'cv_score': result.get('cv_score', tech_score * 100)
            })

            detailed_response = CVResponseGenerator.generate_detailed_response(result)
            result['detailed_analysis'] = detailed_response
            result['analysis'] = detailed_response

            logger.info(f"CV procesado: {result['predicted_class']} (Score: {result.get('cv_score', 0):.1f})")
            return result

        except Exception as e:
            logger.error(f"Error procesando CV {file_path}: {e}")
            return {"error": str(e)}

    def _fallback_analysis(self, tech_score: float, cv_info: dict) -> dict:
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

    def batch_process_cvs(self, cv_folder: str, output_file: str = None) -> list:
        cv_extensions = ['.pdf', '.docx', '.doc', '.txt']
        cv_files = [
            os.path.join(cv_folder, f) for f in os.listdir(cv_folder)
            if any(f.lower().endswith(ext) for ext in cv_extensions)
        ]

        logger.info(f"Encontrados {len(cv_files)} archivos para procesar")

        stats = {'total': len(cv_files), 'apto': 0, 'revisar': 0, 'no_apto': 0, 'errores': 0, 'rechazados_prevalidacion': 0}
        results = []

        for i, cv_file in enumerate(cv_files, 1):
            logger.info(f"Procesando {i}/{len(cv_files)}: {os.path.basename(cv_file)}")
            result = self.process_single_cv(cv_file)
            results.append(result)

            if 'error' in result:
                stats['errores'] += 1
            else:
                cls = result.get('predicted_class', 'Error')
                if cls == 'Apto':
                    stats['apto'] += 1
                elif cls == 'Revisar':
                    stats['revisar'] += 1
                elif cls == 'No apto':
                    stats['no_apto'] += 1
                    if result.get('validation_stage') == 'pre_validation':
                        stats['rechazados_prevalidacion'] += 1

        logger.info("=" * 50)
        logger.info("ESTADÍSTICAS DE PROCESAMIENTO:")
        logger.info(f"Total procesados: {stats['total']}")
        logger.info(f"✅ Aptos: {stats['apto']}")
        logger.info(f"🔍 Revisar: {stats['revisar']}")
        logger.info(f"❌ No aptos: {stats['no_apto']}")
        logger.info(f"   - Rechazados en pre-validación: {stats['rechazados_prevalidacion']}")
        logger.info(f"⚠️  Errores: {stats['errores']}")
        logger.info("=" * 50)

        if output_file:
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