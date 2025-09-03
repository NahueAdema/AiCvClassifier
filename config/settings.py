import os
import re
from typing import List, Set

class Settings:
    # Configuración del modelo
    MAX_SEQUENCE_LENGTH = 512
    EMBEDDING_DIM = 128
    NUM_CLASSES = 3  # Apto, No apto, Revisar
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
    DATA_DIR = os.path.join(BASE_DIR, 'data_samples')

    # Habilidades técnicas más específicas
    TECH_SKILLS = [
        'python', 'java', 'javascript', 'react', 'tensorflow', 'sql',
        'aws', 'docker', 'kubernetes', 'git', 'machine learning',
        'data science', 'html', 'css', 'angular', 'vue', 'nodejs',
        'django', 'flask', 'mongodb', 'postgresql', 'redis', 'jenkins',
        'ci/cd', 'devops', 'microservices', 'api', 'rest', 'graphql',
        'typescript', 'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin',
        'go', 'rust', 'scala', 'r', 'matlab', 'tableau', 'powerbi',
        'spark', 'hadoop', 'kafka', 'elasticsearch', 'linux', 'bash'
    ]

    # Palabras que indican roles técnicos
    TECH_ROLES = [
        'desarrollador', 'developer', 'programador', 'programmer',
        'ingeniero de software', 'software engineer', 'data scientist',
        'analista de datos', 'data analyst', 'devops', 'sre',
        'arquitecto de software', 'tech lead', 'full stack',
        'frontend', 'backend', 'machine learning engineer',
        'ai engineer', 'cloud engineer', 'systems engineer'
    ]

    # Contextos técnicos que dan peso adicional
    TECH_CONTEXTS = [
        'desarrollo de software', 'software development',
        'ciencia de datos', 'data science', 'inteligencia artificial',
        'artificial intelligence', 'machine learning', 'deep learning',
        'web development', 'desarrollo web', 'mobile development',
        'desarrollo móvil', 'cloud computing', 'computación en la nube'
    ]

    SOFT_SKILLS = [
        'liderazgo', 'comunicación', 'trabajo en equipo', 'creatividad',
        'resolución de problemas', 'adaptabilidad', 'organización',
        'gestión de proyectos', 'project management'
    ]

    # Sectores/trabajos que NO son técnicos y deben ser rechazados
    NON_TECH_SECTORS = [
        'supermercado', 'supermarket', 'retail', 'ventas', 'sales',
        'cajero', 'cashier', 'mesero', 'waiter', 'cocina', 'kitchen',
        'limpieza', 'cleaning', 'seguridad', 'security', 'guardia',
        'recepcionista', 'receptionist', 'secretaria', 'secretary',
        'administrativo básico', 'basic admin', 'call center',
        'telemarketing', 'marketing telefónico', 'delivery',
        'repartidor', 'conductor', 'driver', 'almacén básico',
        'warehouse basic', 'empaque', 'packaging', 'operario',
        'factory worker', 'construcción básica', 'basic construction'
    ]

    # Títulos/roles técnicos que sí son válidos
    VALID_TECH_TITLES = [
        'ingeniero', 'engineer', 'desarrollador', 'developer',
        'programador', 'programmer', 'analista', 'analyst',
        'científico de datos', 'data scientist', 'arquitecto',
        'architect', 'especialista en', 'specialist in'
    ]

    # Pesos para scoring (ajustados)
    EXPERIENCE_WEIGHT = 0.25
    TECH_SKILLS_WEIGHT = 0.50  # Aumentado - más importante
    SOFT_SKILLS_WEIGHT = 0.05  # Reducido
    EDUCATION_WEIGHT = 0.15
    LANGUAGES_WEIGHT = 0.05

    # Umbrales más estrictos
    MIN_TECH_SKILLS_REQUIRED = 2  # Mínimo 2 skills técnicas
    MIN_EXPERIENCE_YEARS = 0.5    # Mínimo 6 meses de experiencia
    APTO_THRESHOLD = 0.65         # Umbral más alto para ser "Apto"
    NO_APTO_THRESHOLD = 0.35      # Umbral para "No Apto"

    # Configuración API
    API_HOST = "0.0.0.0"
    API_PORT = 8000

    @classmethod
    def get_all_skills(cls) -> List[str]:
        return cls.TECH_SKILLS + cls.SOFT_SKILLS

    @classmethod
    def extract_tech_skills(cls, text: str) -> Set[str]:
        """Extrae skills técnicas encontradas en el texto."""
        text_lower = text.lower()
        found_skills = set()
        
        for skill in cls.TECH_SKILLS:
            # Búsqueda más precisa con límites de palabra
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        
        return found_skills

    @classmethod
    def has_tech_context(cls, text: str) -> bool:
        """Verifica si el CV tiene contexto técnico relevante."""
        text_lower = text.lower()
        
        # Verificar roles técnicos
        for role in cls.TECH_ROLES:
            if role.lower() in text_lower:
                return True
        
        # Verificar contextos técnicos
        for context in cls.TECH_CONTEXTS:
            if context.lower() in text_lower:
                return True
        
        return False

    @classmethod
    def is_non_tech_profile(cls, text: str) -> bool:
        """Identifica perfiles claramente no técnicos."""
        text_lower = text.lower()
        
        # Contar menciones de sectores no técnicos
        non_tech_mentions = 0
        for sector in cls.NON_TECH_SECTORS:
            if sector.lower() in text_lower:
                non_tech_mentions += 1
        
        # Si tiene muchas menciones no técnicas y pocas técnicas, rechazar
        tech_skills_count = len(cls.extract_tech_skills(text))
        
        return (non_tech_mentions >= 2 and tech_skills_count == 0) or \
               (non_tech_mentions >= 1 and tech_skills_count == 0 and 
                not cls.has_tech_context(text))

    @classmethod
    def has_sufficient_tech_skills(cls, text: str) -> bool:
        """Verifica si el CV tiene suficientes skills técnicas."""
        tech_skills = cls.extract_tech_skills(text)
        return len(tech_skills) >= cls.MIN_TECH_SKILLS_REQUIRED

    @classmethod
    def validate_profile(cls, text: str) -> str:
        """
        Validación previa antes de pasar al scoring.
        Filtro más estricto para rechazar perfiles no técnicos.
        """
        # 1. Verificar si es claramente no técnico
        if cls.is_non_tech_profile(text):
            return "No apto"
        
        # 2. Verificar skills técnicas mínimas
        tech_skills = cls.extract_tech_skills(text)
        if len(tech_skills) == 0:
            return "No apto"
        
        # 3. Si tiene pocas skills técnicas, verificar contexto
        if len(tech_skills) < cls.MIN_TECH_SKILLS_REQUIRED:
            if not cls.has_tech_context(text):
                return "No apto"
        
        # 4. Si pasa los filtros básicos, continuar con scoring
        return "Revisar"  # El modelo luego ajusta entre Apto/Revisar

    @classmethod
    def calculate_tech_score(cls, text: str) -> float:
        """Calcula un score técnico basado en skills y contexto."""
        tech_skills = cls.extract_tech_skills(text)
        has_context = cls.has_tech_context(text)
        
        # Score base por skills
        skills_score = min(len(tech_skills) / 5.0, 1.0)  # Normalizado a max 5 skills
        
        # Bonus por contexto técnico
        context_bonus = 0.2 if has_context else 0.0
        
        # Penalización por sectores no técnicos
        text_lower = text.lower()
        non_tech_penalty = 0.0
        for sector in cls.NON_TECH_SECTORS:
            if sector.lower() in text_lower:
                non_tech_penalty += 0.1
        
        final_score = skills_score + context_bonus - non_tech_penalty
        return max(0.0, min(1.0, final_score))

    @classmethod
    def debug_profile_analysis(cls, text: str) -> dict:
        """Análisis detallado para debugging."""
        return {
            'tech_skills_found': list(cls.extract_tech_skills(text)),
            'tech_skills_count': len(cls.extract_tech_skills(text)),
            'has_tech_context': cls.has_tech_context(text),
            'is_non_tech': cls.is_non_tech_profile(text),
            'tech_score': cls.calculate_tech_score(text),
            'validation_result': cls.validate_profile(text)
        }


settings = Settings()