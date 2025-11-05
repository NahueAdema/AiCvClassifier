import logging
from config.settings import Settings
from config.responses import CVResponseGenerator

logger = logging.getLogger(__name__)


class CVValidator:
    @staticmethod
    def quick_validate(text: str) -> dict:
        validation_result = Settings.validate_profile(text)
        debug_analysis = Settings.debug_profile_analysis(text)
        tech_score = Settings.calculate_tech_score(text)

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
            'recommendation': CVValidator._get_recommendation(validation_result, tech_score),
            'detailed_assessment': detailed_response,
            'quick_analysis': detailed_response
        }

    @staticmethod
    def _get_recommendation(validation: str, tech_score: float) -> str:
        if validation == "No apto":
            return "❌ Rechazar - No cumple criterios técnicos mínimos"
        elif tech_score >= 0.7:
            return "✅ Procesar - Perfil técnico sólido"
        elif tech_score >= 0.4:
            return "🔍 Revisar - Perfil técnico medio, requiere evaluación detallada"
        else:
            return "⚠️  Cuidado - Score técnico bajo, revisar cuidadosamente"

    @staticmethod
    def is_technical_profile(text: str) -> bool:
        return Settings.validate_profile(text) != "No apto"

    @staticmethod
    def calculate_tech_score(text: str) -> float:
        return Settings.calculate_tech_score(text)

    @staticmethod
    def get_rejection_result(file_path: str, text: str, reason: str, stage: str, confidence: float = 0.95) -> dict:
        tech_score = CVValidator.calculate_tech_score(text)
        debug_analysis = Settings.debug_profile_analysis(text)
        cv_info = {'skills': [], 'experience_years': 0}

        basic_result = {
            'predicted_class': 'No apto',
            'confidence': confidence,
            'cv_score': 0.0,
            'tech_score': tech_score,
            'cv_info': cv_info,
            'rejection_reason': reason,
            'debug_info': debug_analysis,
            'text_preview': text[:200] + "..." if len(text) > 200 else text,
            'file_path': file_path,
            'validation_stage': stage
        }

        detailed_response = CVResponseGenerator.generate_detailed_response(basic_result)
        basic_result['detailed_analysis'] = detailed_response
        basic_result['analysis'] = detailed_response
        return basic_result