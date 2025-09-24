# config/responses.py
from typing import Dict, List
import random

class CVResponseGenerator:
    """Genera respuestas detalladas y personalizadas para el análisis de CVs"""
    
    # Plantillas de respuestas por categoría
    RESPONSE_TEMPLATES = {
        "apto": {
            "titles": [
                "🎯 Perfil Técnico Sólido Detectado",
                "✅ Candidato Ideal para Posiciones Técnicas",
                "🌟 Excelente Perfil para Desarrollo de Software"
            ],
            "strengths": [
                "Destaca por su experiencia en {strengths}",
                "Fuerte conocimiento en tecnologías como {strengths}",
                "Competencias técnicas sólidas en {strengths}"
            ],
            "recommendations": [
                "Programar entrevista técnica lo antes posible",
                "Considerar para posiciones senior o de liderazgo técnico",
                "Evaluar para proyectos específicos de {top_skills}"
            ],
            "next_steps": [
                "Contactar dentro de 24-48 horas",
                "Preparar evaluación técnica avanzada",
                "Coordinar entrevista con el equipo técnico"
            ]
        },
        
        "revisar": {
            "titles": [
                "🔍 Perfil que Merece Revisión Detallada",
                "📊 Candidato con Potencial por Desarrollar",
                "⚠️ Requiere Evaluación Adicional"
            ],
            "strengths": [
                "Bases técnicas presentes en {strengths}",
                "Conocimientos iniciales en {strengths}",
                "Potencial detectable en áreas como {strengths}"
            ],
            "concerns": [
                "Falta experiencia en {missing_areas}",
                "Habilidades técnicas limitadas para posiciones senior",
                "Necesita desarrollo en {improvement_areas}"
            ],
            "recommendations": [
                "Realizar entrevista de screening técnico",
                "Evaluar con prueba técnica básica",
                "Considerar para posiciones junior o de entrenamiento"
            ]
        },
        
        "no_apto": {
            "titles": [
                "❌ Perfil No Alineado con Requisitos Técnicos",
                "🚫 Candidato para Otras Áreas No Técnicas",
                "📉 Perfil No Compatible con Posiciones de Desarrollo"
            ],
            "reasons": [
                "Falta de habilidades técnicas específicas",
                "Experiencia en áreas no relacionadas con tecnología",
                "Perfil orientado a {current_sector} en lugar de desarrollo"
            ],
            "suggestions": [
                "Considerar para posiciones no técnicas en la empresa",
                "Reevaluar si busca transición a tecnología con capacitación",
                "Archivar para futuras posiciones no técnicas"
            ]
        }
    }
    
    # Mensajes específicos por habilidades
    SKILL_COMMENTS = {
        "python": "Python es fundamental para desarrollo backend y data science",
        "javascript": "JavaScript esencial para desarrollo web full-stack",
        "react": "React demuestra experiencia en frontend moderno",
        "java": "Java indica experiencia en sistemas empresariales",
        "sql": "SQL crucial para manejo de bases de datos",
        "aws": "AWS muestra conocimiento en cloud computing",
        "docker": "Docker evidencia experiencia en DevOps",
        "machine learning": "ML indica especialización en inteligencia artificial"
    }
    
    # Recomendaciones por nivel de experiencia
    EXPERIENCE_RECOMMENDATIONS = {
        "junior": "Ideal para posiciones de entrenamiento o junior",
        "mid": "Apto para roles de desarrollador regular",
        "senior": "Considerar para liderazgo técnico o arquitectura",
        "lead": "Excelente para gestión de equipos o proyectos complejos"
    }
    
    @classmethod
    def generate_detailed_response(cls, analysis_result: Dict) -> Dict:
        """Genera una respuesta detallada y personalizada"""
        
        # Determinar categoría principal
        category = analysis_result['predicted_class'].lower().replace(" ", "_")
        if "apto" in category:
            category_key = "apto"
        elif "revisar" in category:
            category_key = "revisar"
        else:
            category_key = "no_apto"
        
        # Extraer información del análisis
        cv_info = analysis_result.get('cv_info', {})
        skills = cv_info.get('skills', [])
        experience = cv_info.get('experience_years', 0)
        education = cv_info.get('education', '')
        tech_score = analysis_result.get('tech_score', 0)
        confidence = analysis_result.get('confidence', 0)
        
        # Generar elementos de la respuesta
        title = cls._generate_title(category_key, tech_score, confidence)
        strengths = cls._analyze_strengths(skills, experience, education)
        concerns = cls._identify_concerns(skills, experience, category_key)
        recommendations = cls._generate_recommendations(category_key, skills, experience, analysis_result)
        technical_breakdown = cls._create_technical_breakdown(analysis_result)
        
        return {
            "title": title,
            "category": category_key,
            "confidence_score": round(confidence * 100, 1),
            "technical_score": round(tech_score * 100, 1),
            "executive_summary": cls._generate_executive_summary(category_key, strengths, tech_score),
            "strengths_analysis": strengths,
            "areas_for_improvement": concerns,
            "technical_breakdown": technical_breakdown,
            "recommendations": recommendations,
            "next_steps": cls._generate_next_steps(category_key, confidence),
            "detailed_assessment": cls._generate_detailed_assessment(analysis_result)
        }
    
    @classmethod
    def _generate_title(cls, category: str, tech_score: float, confidence: float) -> str:
        """Genera título personalizado según categoría y scores"""
        templates = cls.RESPONSE_TEMPLATES[category]["titles"]
        base_title = random.choice(templates)
        
        if category == "apto" and tech_score > 0.8:
            return f"🎯 {base_title} - Perfil Destacado"
        elif category == "revisar" and confidence < 0.7:
            return f"⚠️ {base_title} - Decisión Borderline"
        else:
            return base_title
    
    @classmethod
    def _analyze_strengths(cls, skills: List[str], experience: int, education: str) -> Dict:
        """Analiza y describe las fortalezas del candidato"""
        tech_skills = [skill for skill in skills if skill.lower() in cls.SKILL_COMMENTS]
        strength_comments = []
        
        for skill in tech_skills[:5]:  # Top 5 habilidades
            if skill.lower() in cls.SKILL_COMMENTS:
                strength_comments.append(cls.SKILL_COMMENTS[skill.lower()])
        
        # Determinar nivel de experiencia
        if experience >= 5:
            exp_level = "senior"
        elif experience >= 2:
            exp_level = "mid"
        else:
            exp_level = "junior"
        
        return {
            "technical_skills": {
                "count": len(tech_skills),
                "top_skills": tech_skills[:5],
                "comments": strength_comments
            },
            "experience": {
                "years": experience,
                "level": exp_level,
                "assessment": cls.EXPERIENCE_RECOMMENDATIONS[exp_level]
            },
            "education": {
                "level": education,
                "relevance": "Técnica" if education in ['bachelor', 'master', 'doctorate', 'technical'] else "No técnica"
            }
        }
    
    @classmethod
    def _identify_concerns(cls, skills: List[str], experience: int, category: str) -> List[str]:
        """Identifica áreas de mejora"""
        concerns = []
        
        if category == "no_apto":
            if len(skills) == 0:
                concerns.append("Falta completa de habilidades técnicas identificables")
            if experience == 0:
                concerns.append("Sin experiencia profesional demostrable")
            else:
                concerns.append("Experiencia en sectores no relacionados con tecnología")
        
        elif category == "revisar":
            if len(skills) < 3:
                concerns.append("Habilidades técnicas limitadas para posiciones competitivas")
            if experience < 2:
                concerns.append("Experiencia insuficiente para roles mid-level")
        
        return concerns if concerns else ["No se identificaron preocupaciones mayores"]
    
    @classmethod
    def _generate_recommendations(cls, category: str, skills: List[str], experience: int, analysis: Dict) -> List[Dict]:
        """Genera recomendaciones específicas"""
        recommendations = []
        
        if category == "apto":
            recommendations.append({
                "priority": "Alta",
                "action": "Entrevista técnica inmediata",
                "reason": "Perfil altamente compatible con posiciones técnicas"
            })
            
            if experience >= 3:
                recommendations.append({
                    "priority": "Media",
                    "action": "Evaluar para liderazgo técnico",
                    "reason": "Experiencia suficiente para roles senior"
                })
                
        elif category == "revisar":
            recommendations.append({
                "priority": "Media",
                "action": "Screening técnico inicial",
                "reason": "Validar habilidades técnicas básicas"
            })
            
            if len(skills) > 0:
                recommendations.append({
                    "priority": "Baja",
                    "action": "Prueba técnica específica",
                    "reason": f"Evaluar profundidad en {skills[0]}"
                })
                
        else:  # no_apto
            recommendations.append({
                "priority": "Baja",
                "action": "Archivar para futuras referencias",
                "reason": "Perfil no alineado con necesidades técnicas actuales"
            })
        
        return recommendations
    
    @classmethod
    def _create_technical_breakdown(cls, analysis: Dict) -> Dict:
        """Crea un breakdown técnico detallado"""
        cv_info = analysis.get('cv_info', {})
        
        return {
            "skills_analysis": {
                "total_skills": len(cv_info.get('skills', [])),
                "tech_skills_ratio": cls._calculate_tech_ratio(cv_info.get('skills', [])),
                "skill_diversity": cls._assess_skill_diversity(cv_info.get('skills', []))
            },
            "experience_quality": {
                "years": cv_info.get('experience_years', 0),
                "assessment": cls._assess_experience_quality(cv_info.get('experience_years', 0))
            },
            "completeness_score": cls._calculate_completeness(cv_info),
            "red_flags": cls._identify_red_flags(analysis)
        }
    
    @classmethod
    def _generate_executive_summary(cls, category: str, strengths: Dict, tech_score: float) -> str:
        """Genera un resumen ejecutivo"""
        skill_count = strengths["technical_skills"]["count"]
        exp_level = strengths["experience"]["level"]
        
        if category == "apto":
            return f"Candidato con {skill_count} habilidades técnicas sólidas y perfil {exp_level}. Score técnico: {tech_score*100:.1f}% - Recomendado para proceso inmediato."
        elif category == "revisar":
            return f"Perfil con {skill_count} habilidades técnicas identificadas. Nivel {exp_level}. Score técnico: {tech_score*100:.1f}% - Requiere evaluación adicional."
        else:
            return f"Perfil no técnico identificado. Score técnico: {tech_score*100:.1f}% - No recomendado para posiciones técnicas."
    
    @classmethod
    def _generate_next_steps(cls, category: str, confidence: float) -> List[str]:
        """Genera próximos pasos específicos"""
        if category == "apto":
            if confidence > 0.8:
                return ["Contactar dentro de 24 horas", "Programar entrevista técnica", "Preparar oferta potencial"]
            else:
                return ["Contactar dentro de 48 horas", "Validar referencias", "Entrevista de screening"]
        elif category == "revisar":
            return ["Revisar en 1-2 semanas", "Comparar con otros candidatos", "Evaluar necesidad de capacitación"]
        else:
            return ["Archivar candidatura", "Considerar para posiciones no técnicas futuras"]
    
    @classmethod
    def _generate_detailed_assessment(cls, analysis: Dict) -> str:
        """Genera una evaluación detallada en texto natural"""
        cv_info = analysis.get('cv_info', {})
        skills = cv_info.get('skills', [])
        experience = cv_info.get('experience_years', 0)
        
        assessment_parts = []
        
        # Análisis de habilidades
        if skills:
            assessment_parts.append(f"El candidato demuestra conocimiento en {len(skills)} áreas técnicas, destacando en {', '.join(skills[:3])}.")
        else:
            assessment_parts.append("No se identificaron habilidades técnicas específicas en el CV.")
        
        # Análisis de experiencia
        if experience > 0:
            assessment_parts.append(f"Cuenta con {experience} años de experiencia profesional.")
        else:
            assessment_parts.append("Perfil junior o recién graduado sin experiencia laboral extensa.")
        
        # Score técnico
        tech_score = analysis.get('tech_score', 0)
        assessment_parts.append(f"El análisis técnico arrojó un score de {tech_score*100:.1f}%.")
        
        return " ".join(assessment_parts)
    
    # Métodos auxiliares
    @classmethod
    def _calculate_tech_ratio(cls, skills: List[str]) -> float:
        from config.settings import Settings
        tech_skills = [s for s in skills if s.lower() in [ts.lower() for ts in Settings.TECH_SKILLS]]
        return len(tech_skills) / len(skills) if skills else 0
    
    @classmethod
    def _assess_skill_diversity(cls, skills: List[str]) -> str:
        categories = set()
        for skill in skills:
            if any(tech in skill.lower() for tech in ['python', 'java', 'c++', 'javascript']):
                categories.add('programming')
            elif any(tech in skill.lower() for tech in ['sql', 'database', 'mysql']):
                categories.add('database')
            elif any(tech in skill.lower() for tech in ['react', 'angular', 'vue', 'frontend']):
                categories.add('frontend')
            elif any(tech in skill.lower() for tech in ['aws', 'azure', 'cloud', 'devops']):
                categories.add('infrastructure')
        
        return f"Diversidad en {len(categories)} áreas técnicas" if categories else "Habilidades limitadas a pocas áreas"
    
    @classmethod
    def _assess_experience_quality(cls, experience_years: int) -> str:
        if experience_years >= 5:
            return "Experiencia sólida y diversa"
        elif experience_years >= 2:
            return "Experiencia suficiente para roles mid-level"
        elif experience_years > 0:
            return "Experiencia inicial/junior"
        else:
            return "Sin experiencia profesional"
    
    @classmethod
    def _calculate_completeness(cls, cv_info: Dict) -> int:
        score = 0
        if cv_info.get('email'): score += 25
        if cv_info.get('phone'): score += 25
        if cv_info.get('skills'): score += 25
        if cv_info.get('experience_years', 0) > 0: score += 25
        return score
    
    @classmethod
    def _identify_red_flags(cls, analysis: Dict) -> List[str]:
        red_flags = []
        cv_info = analysis.get('cv_info', {})
        
        if not cv_info.get('email'):
            red_flags.append("Falta información de contacto (email)")
        if not cv_info.get('skills'):
            red_flags.append("No se identificaron habilidades técnicas")
        if analysis.get('tech_score', 0) < 0.3:
            red_flags.append("Score técnico muy bajo")
            
        return red_flags