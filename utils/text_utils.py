import re
import string
from typing import List, Dict, Set
import unicodedata

def normalize_text(text: str) -> str:
    """Normaliza texto removiendo acentos y caracteres especiales"""
    # Normalizar unicode
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text.lower()

def extract_years_experience(text: str) -> List[int]:
    """Extrae años de experiencia mencionados en el texto"""
    patterns = [
        r'(\d+)\s*(?:años?|years?)\s*(?:de\s*)?(?:experiencia|experience)',
        r'(?:experiencia|experience).*?(\d+)\s*(?:años?|years?)',
        r'(\d+)\+?\s*(?:años?|years?)',
        r'(?:con\s+)?(\d+)\s*(?:años?|years?)\s*(?:en|in|de|of)',
        r'(\d+)\s*(?:años?|years?)\s*(?:como|as|trabajando|working)'
    ]
    
    years = []
    text_lower = text.lower()
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if match.isdigit():
                year_value = int(match)
                # Filtrar valores poco realistas
                if 0 <= year_value <= 50:
                    years.append(year_value)
    
    return sorted(set(years), reverse=True)

def extract_programming_languages(text: str) -> Set[str]:
    """Extrae lenguajes de programación del texto con mayor precisión"""
    languages = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'cpp',
        'c#', 'csharp', 'php', 'ruby', 'swift', 'kotlin', 'go', 'rust',
        'scala', 'r', 'matlab', 'perl', 'shell', 'bash', 'powershell',
        'dart', 'elixir', 'haskell', 'clojure', 'fortran', 'cobol'
    }
    
    found_languages = set()
    text_lower = text.lower()
    
    # Patrones más específicos para cada lenguaje
    language_patterns = {
        'python': r'\bpython\b',
        'java': r'\bjava\b(?!\s*script)',  # Java pero no JavaScript
        'javascript': r'\b(?:javascript|js)\b',
        'typescript': r'\b(?:typescript|ts)\b',
        'c++': r'\bc\+\+\b',
        'cpp': r'\bcpp\b',
        'c#': r'\bc#\b',
        'csharp': r'\bc\s*sharp\b',
        'php': r'\bphp\b',
        'ruby': r'\bruby\b',
        'swift': r'\bswift\b',
        'kotlin': r'\bkotlin\b',
        'go': r'\bgo(?:lang)?\b',
        'rust': r'\brust\b',
        'scala': r'\bscala\b',
        'r': r'\b(?:r\s+programming|r\s+language|\br\b(?:\s+(?:lang|programming)))',
        'matlab': r'\bmatlab\b',
        'perl': r'\bperl\b',
        'shell': r'\b(?:shell|bash)\b',
        'powershell': r'\bpowershell\b'
    }
    
    for lang, pattern in language_patterns.items():
        if re.search(pattern, text_lower):
            found_languages.add(lang)
    
    # También buscar lenguajes sin patrones específicos
    for lang in languages:
        if lang not in language_patterns:
            if re.search(r'\b' + re.escape(lang) + r'\b', text_lower):
                found_languages.add(lang)
    
    return found_languages

def extract_technologies(text: str) -> Set[str]:
    """Extrae tecnologías y frameworks del texto con mayor precisión"""
    technologies = {
        'react', 'angular', 'vue', 'nodejs', 'node.js', 'express', 'django',
        'flask', 'spring', 'laravel', 'rails', 'asp.net', 'tensorflow',
        'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'docker',
        'kubernetes', 'aws', 'azure', 'gcp', 'mysql', 'postgresql',
        'mongodb', 'redis', 'elasticsearch', 'git', 'jenkins', 'ci/cd',
        'hadoop', 'spark', 'kafka', 'rabbitmq', 'nginx', 'apache',
        'terraform', 'ansible', 'helm', 'grafana', 'prometheus'
    }
    
    found_tech = set()
    text_lower = text.lower()
    
    # Patrones específicos para tecnologías que pueden tener falsos positivos
    tech_patterns = {
        'react': r'\breact(?:\s+js|\s+native)?\b',
        'angular': r'\bangular(?:\s+js)?\b',
        'vue': r'\bvue(?:\s+js)?\b',
        'nodejs': r'\b(?:node\.?js|nodejs)\b',
        'spring': r'\bspring(?:\s+boot|\s+framework)?\b',
        'git': r'\bgit\b(?!\s*hub)',  # Git pero no GitHub
        'aws': r'\b(?:aws|amazon\s+web\s+services)\b',
        'gcp': r'\b(?:gcp|google\s+cloud)\b',
        'ci/cd': r'\bci\s*/\s*cd\b'
    }
    
    for tech, pattern in tech_patterns.items():
        if re.search(pattern, text_lower):
            found_tech.add(tech)
    
    # Buscar el resto de tecnologías
    for tech in technologies:
        if tech not in tech_patterns:
            if re.search(r'\b' + re.escape(tech) + r'\b', text_lower):
                found_tech.add(tech)
    
    return found_tech

def is_technical_context(text: str) -> bool:
    """Determina si el texto está en un contexto técnico"""
    technical_indicators = [
        'desarrollo', 'development', 'programación', 'programming',
        'software', 'sistemas', 'systems', 'tecnología', 'technology',
        'ingeniería', 'engineering', 'arquitectura', 'architecture',
        'datos', 'data', 'algoritmo', 'algorithm', 'código', 'code',
        'aplicación', 'application', 'web', 'móvil', 'mobile',
        'base de datos', 'database', 'servidor', 'server',
        'api', 'framework', 'biblioteca', 'library'
    ]
    
    text_lower = text.lower()
    context_count = sum(1 for indicator in technical_indicators 
                       if indicator in text_lower)
    
    return context_count >= 3  # Al menos 3 indicadores técnicos

def extract_non_technical_jobs(text: str) -> Set[str]:
    """Identifica trabajos claramente no técnicos"""
    non_tech_jobs = [
        'cajero', 'cashier', 'vendedor', 'salesperson', 'mesero', 'waiter',
        'cocinero', 'cook', 'chef', 'limpieza', 'cleaning', 'seguridad',
        'security', 'guardia', 'guard', 'recepcionista', 'receptionist',
        'secretaria', 'secretary', 'administrativo', 'admin assistant',
        'call center', 'telemarketing', 'delivery', 'repartidor',
        'conductor', 'driver', 'operario', 'operator', 'almacén',
        'warehouse', 'empaque', 'packaging', 'supermercado', 'supermarket',
        'retail', 'tienda', 'store', 'restaurante', 'restaurant'
    ]
    
    found_jobs = set()
    text_lower = text.lower()
    
    for job in non_tech_jobs:
        if re.search(r'\b' + re.escape(job.lower()) + r'\b', text_lower):
            found_jobs.add(job)
    
    return found_jobs

def calculate_technical_relevance_score(text: str) -> float:
    """Calcula un score de relevancia técnica del CV"""
    # Factores positivos
    prog_langs = extract_programming_languages(text)
    technologies = extract_technologies(text)
    has_tech_context = is_technical_context(text)
    
    # Factores negativos
    non_tech_jobs = extract_non_technical_jobs(text)
    
    # Cálculo del score
    positive_score = 0.0
    positive_score += len(prog_langs) * 0.3  # Lenguajes de programación
    positive_score += len(technologies) * 0.2  # Tecnologías
    positive_score += 0.3 if has_tech_context else 0.0  # Contexto técnico
    
    # Penalizaciones
    negative_score = len(non_tech_jobs) * 0.4  # Trabajos no técnicos
    
    final_score = max(0.0, positive_score - negative_score)
    return min(1.0, final_score)  # Normalizar entre 0 y 1

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calcula similitud simple entre dos textos"""
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def extract_contact_info(text: str) -> Dict[str, str]:
    """Extrae información de contacto del texto"""
    contact_info = {}
    
    # Email con validación más estricta
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        # Filtrar emails que no sean válidos
        valid_emails = [email for email in emails 
                       if not re.search(r'@.*\.(jpg|png|pdf|doc)$', email.lower())]
        if valid_emails:
            contact_info['email'] = valid_emails[0]
    
    # Teléfono con patrones más específicos
    phone_patterns = [
        r'\+?[\d\s\-\(\)]{10,15}',
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        r'\b\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{4}\b'
    ]
    
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        if phones:
            # Limpiar el teléfono de caracteres no numéricos para validar
            clean_phone = re.sub(r'[^\d]', '', phones[0])
            if 7 <= len(clean_phone) <= 15:  # Longitud válida
                contact_info['phone'] = phones[0]
                break
    
    # LinkedIn
    linkedin_pattern = r'linkedin\.com/in/[\w\-]+'
    linkedin_matches = re.findall(linkedin_pattern, text.lower())
    if linkedin_matches:
        contact_info['linkedin'] = f"https://{linkedin_matches[0]}"
    
    # GitHub
    github_pattern = r'github\.com/[\w\-]+'
    github_matches = re.findall(github_pattern, text.lower())
    if github_matches:
        contact_info['github'] = f"https://{github_matches[0]}"
    
    return contact_info

def clean_skill_name(skill: str) -> str:
    """Limpia y normaliza nombres de habilidades"""
    # Remover caracteres especiales excepto algunos importantes
    clean = re.sub(r'[^\w\s\+\#\.]', '', skill)
    # Normalizar espacios
    clean = re.sub(r'\s+', ' ', clean).strip()
    
    # Casos especiales con mapeo más completo
    skill_mappings = {
        'c++': 'cpp',
        'c#': 'csharp',
        'js': 'javascript',
        'ts': 'typescript',
        'ai': 'artificial intelligence',
        'ml': 'machine learning',
        'dl': 'deep learning',
        'node': 'nodejs',
        'react.js': 'react',
        'vue.js': 'vue',
        'angular.js': 'angular'
    }
    
    return skill_mappings.get(clean.lower(), clean.lower())

def extract_education_info(text: str) -> Dict[str, any]:
    """Extrae información educativa del texto con mayor precisión"""
    education_info = {
        'degrees': [],
        'institutions': [],
        'certifications': [],
        'level': 'none'
    }
    
    # Patrones de títulos más específicos
    degree_patterns = [
        (r'(?:licenciatura|bachelor|b\.?[as]\.?)\s+(?:en\s+)?([a-zA-Z\s]+)', 'bachelor'),
        (r'(?:maestría|master|m\.?[as]\.?)\s+(?:en\s+)?([a-zA-Z\s]+)', 'master'),
        (r'(?:doctorado|doctorate|phd|ph\.?d\.?)\s+(?:en\s+)?([a-zA-Z\s]+)', 'doctorate'),
        (r'(?:ingeniería|engineering)\s+(?:en\s+)?([a-zA-Z\s]+)', 'bachelor'),
        (r'(?:técnico|technical|technician)\s+(?:en\s+)?([a-zA-Z\s]+)', 'technical')
    ]
    
    text_lower = text.lower()
    for pattern, level in degree_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            education_info['degrees'].extend(matches)
            if level == 'doctorate':
                education_info['level'] = 'doctorate'
            elif level == 'master' and education_info['level'] not in ['doctorate']:
                education_info['level'] = 'master'
            elif level == 'bachelor' and education_info['level'] not in ['doctorate', 'master']:
                education_info['level'] = 'bachelor'
            elif level == 'technical' and education_info['level'] == 'none':
                education_info['level'] = 'technical'
    
    # Certificaciones técnicas
    cert_patterns = [
        r'(aws\s+certified[^,\n]*)',
        r'(google\s+certified[^,\n]*)',
        r'(microsoft\s+certified[^,\n]*)',
        r'(oracle\s+certified[^,\n]*)',
        r'(cisco\s+certified[^,\n]*)',
        r'(pmp\s+certified?)',
        r'(scrum\s+master)',
        r'(cissp|ceh|comptia[^,\n]*)'
    ]
    
    for pattern in cert_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        education_info['certifications'].extend(matches)
    
    return education_info

def calculate_keyword_density(text: str, keywords: List[str]) -> Dict[str, float]:
    """Calcula la densidad de palabras clave en el texto"""
    text_lower = normalize_text(text)
    words = text_lower.split()
    word_count = len(words)
    
    densities = {}
    for keyword in keywords:
        keyword_lower = normalize_text(keyword)
        keyword_count = text_lower.count(keyword_lower)
        densities[keyword] = keyword_count / word_count if word_count > 0 else 0.0
    
    return densities

def detect_cv_language(text: str) -> str:
    """Detecta el idioma principal del CV"""
    spanish_indicators = ['años', 'experiencia', 'trabajo', 'empresa', 'universidad', 'licenciatura']
    english_indicators = ['years', 'experience', 'work', 'company', 'university', 'bachelor']
    
    text_lower = text.lower()
    spanish_count = sum(1 for indicator in spanish_indicators if indicator in text_lower)
    english_count = sum(1 for indicator in english_indicators if indicator in text_lower)
    
    return 'spanish' if spanish_count > english_count else 'english'