# config/feedback_messages.py
POSITIVE_FEEDBACK = [
    "Dominio claro de {tech_skills_list} en entornos reales",
    "Experiencia alineada con roles de {tech_roles_found}",
    "Mención de contextos técnicos relevantes: {tech_contexts_found}",
    "Perfil técnico sólido con {experience_years}+ años en el sector",
    "Uso adecuado de terminología técnica en áreas como {relevant_domains}"
]

CONSTRUCTIVE_FEEDBACK = [
    "Detectamos skills técnicas ({detected_skills}), pero faltan al menos {missing_count} más para ser competitivo",
    "Tu experiencia en {detected_roles} es valiosa; mejoralo con logros medibles (ej: 'optimicé X en un Y%')",
    "No encontramos menciones a herramientas clave como {missing_critical_tools}",
    "El CV es genérico; enfócate en destacar proyectos con {suggested_focus}",
    "Considerá añadir contexto técnico: ¿participaste en desarrollo ágil, CI/CD, cloud, etc.?"
]

ACTIONABLE_ADVICE = [
    "Añadí al menos 2-3 de estas skills en tu próximo CV: {top_missing_skills}",
    "Incluí un proyecto con {missing_context} (ej: 'API REST con FastAPI y Docker')",
    "Mencioná logros con números: 'reduje tiempos en un 30%', 'manejé X usuarios', etc.",
    "Certificaciones recomendadas: {recommended_certs} (ej: AWS Cloud Practitioner, Google Data Analytics)",
    "Reescribí tu resumen para incluir: {tech_role} con experiencia en {key_skills}"
]

REJECTION_FEEDBACK = [
    "El perfil no muestra suficientes señales técnicas (solo {skill_count} skills detectadas)",
    "Predominan roles no técnicos ({non_tech_roles}); considerá reescribir enfocado en tecnología",
    "No se identifican contextos de desarrollo de software, data science o AI",
    "Recomendamos formación técnica adicional antes de postular a roles técnicos",
    "Tu CV parece orientado a {non_tech_sector}; para tecnología, necesitás resaltar {required_elements}"
]