import fitz  # PyMuPDF
from docx import Document
import re
from typing import Dict, Optional
import logging
 
logger = logging.getLogger(__name__)

class DocumentExtractor:
    """Extrae texto de documentos PDF, Word y texto plano"""
    
    def extract_text(self, file_path: str) -> Optional[str]:
        """Extrae texto basado en la extensión del archivo"""
        try:
            if file_path.lower().endswith('.pdf'):
                return self._extract_pdf(file_path)
            elif file_path.lower().endswith(('.docx', '.doc')):
                return self._extract_word(file_path)
            elif file_path.lower().endswith('.txt'):
                return self._extract_txt(file_path)
            else:
                logger.error(f"Formato no soportado: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error extrayendo texto de {file_path}: {e}")
            return None
    
    def _extract_pdf(self, file_path: str) -> str:
        """Extrae texto de PDF usando PyMuPDF"""
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return self._clean_text(text)
    
    def _extract_word(self, file_path: str) -> str:
        """Extrae texto de documentos Word"""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return self._clean_text(text)
    
    def _extract_txt(self, file_path: str) -> str:
        """Extrae texto de archivos de texto plano"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
        return self._clean_text(text)
    
    def _clean_text(self, text: str) -> str:
        """Limpia y normaliza el texto extraído"""
        # Remover caracteres especiales excesivos
        text = re.sub(r'\s+', ' ', text)  # Múltiples espacios a uno solo
        text = re.sub(r'\n+', '\n', text)  # Múltiples saltos de línea
        text = text.strip()
        return text

class CVInfoExtractor:
    """Extrae información específica del CV como email, teléfono, habilidades"""
    
    def __init__(self):
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.phone_pattern = r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        
    def extract_info(self, text: str) -> Dict[str, any]:
        """Extrae información estructurada del texto del CV"""
        info = {
            'email': self._extract_email(text),
            'phone': self._extract_phone(text),
            'skills': self._extract_skills(text),
            'experience_years': self._estimate_experience(text),
            'education': self._extract_education_level(text)
        }
        return info
    
    def _extract_email(self, text: str) -> Optional[str]:
        emails = re.findall(self.email_pattern, text)
        return emails[0] if emails else None
    
    def _extract_phone(self, text: str) -> Optional[str]:
        phones = re.findall(self.phone_pattern, text)
        return phones[0] if phones else None
    
    def _extract_skills(self, text: str) -> list:
        from config.settings import settings
        text_lower = text.lower()
        found_skills = []
        
        for skill in settings.get_all_skills():
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def _estimate_experience(self, text: str) -> int:
        """Estima años de experiencia basado en patrones en el texto"""
        patterns = [
            r'(\d+)\s*(?:años?|years?)\s*(?:de\s*)?(?:experiencia|experience)',
            r'(?:experiencia|experience).*?(\d+)\s*(?:años?|years?)',
        ]
        
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            years.extend([int(match) for match in matches])
        
        return max(years) if years else 0
    
    def _extract_education_level(self, text: str) -> str:
        """Determina el nivel de educación más alto"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['doctorado', 'phd', 'doctorate']):
            return 'doctorate'
        elif any(word in text_lower for word in ['maestría', 'master', 'mba']):
            return 'master'
        elif any(word in text_lower for word in ['licenciatura', 'ingeniería', 'bachelor']):
            return 'bachelor'
        elif any(word in text_lower for word in ['técnico', 'technician']):
            return 'technical'
        else:
            return 'other'