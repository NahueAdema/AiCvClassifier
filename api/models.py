from pydantic import BaseModel
from typing import List, Dict, Optional

class JobRequirements(BaseModel):
    description: str
    required_skills: List[str]
    min_experience: int = 0
    preferred_education: str = 'bachelor'

class CVAnalysisResponse(BaseModel):
    predicted_class: str
    confidence: float
    cv_score: float
    cv_info: Dict
    recommendation: Optional[str] = None
    missing_skills: Optional[List[str]] = None

class BatchProcessRequest(BaseModel):
    job_requirements: Optional[JobRequirements] = None

class ValidateTextRequest(BaseModel):
    text: str