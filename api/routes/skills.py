from fastapi import APIRouter, HTTPException
from api.models import JobRequirements
from api.deps import get_cv_app
from config.settings import Settings
import logging
import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/skills-available")
async def get_available_skills():
    try:
        return {
            "tech_skills": Settings.TECH_SKILLS,
            "soft_skills": Settings.SOFT_SKILLS,
            "tech_roles": Settings.TECH_ROLES,
            "non_tech_sectors": Settings.NON_TECH_SECTORS,
            "counts": {
                "tech_skills": len(Settings.TECH_SKILLS),
                "soft_skills": len(Settings.SOFT_SKILLS),
                "tech_roles": len(Settings.TECH_ROLES),
                "non_tech_sectors": len(Settings.NON_TECH_SECTORS)
            },
            "all_skills": Settings.get_all_skills()
        }
    except Exception as e:
        logger.error(f"Error obteniendo skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-job-profile")
async def create_job_profile(job_req: JobRequirements):
    try:
        cv_app = get_cv_app()
        profile = cv_app.create_job_profile(
            job_req.description,
            job_req.required_skills,
            job_req.min_experience,
            job_req.preferred_education
        )
        return {
            "job_profile": profile,
            "status": "created",
            "created_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creando perfil: {e}")
        raise HTTPException(status_code=500, detail=str(e))