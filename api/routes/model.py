from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.deps import get_cv_app
from core.trainer import ModelTrainer  
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/model-status")
async def model_status():
    from config.settings import Settings
    cv_app = get_cv_app()
    model_path = os.path.join(Settings.MODEL_DIR, 'cv_classifier_model')
    model_exists = os.path.exists(model_path)

    status = {
        "model_trained": model_exists,
        "classifier_ready": cv_app.classifier.model is not None,
        "settings": {
            "min_tech_skills": Settings.MIN_TECH_SKILLS_REQUIRED,
            "apto_threshold": Settings.APTO_THRESHOLD,
            "no_apto_threshold": Settings.NO_APTO_THRESHOLD,
            "max_sequence_length": Settings.MAX_SEQUENCE_LENGTH
        },
        "system_info": {
            "tech_skills_count": len(Settings.TECH_SKILLS),
            "soft_skills_count": len(Settings.SOFT_SKILLS),
        },
        "checked_at": datetime.now().isoformat()
    }

    if model_exists:
        try:
            model_stat = os.stat(model_path)
            status["model_info"] = {
                "created_at": datetime.fromtimestamp(model_stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(model_stat.st_mtime).isoformat(),
                "size_mb": round(model_stat.st_size / (1024 * 1024), 2)
            }
            if not cv_app.classifier.model:
                cv_app.classifier.load_model(model_path)
            status["model_loaded"] = cv_app.classifier.model is not None
        except Exception as e:
            logger.warning(f"Error cargando modelo: {e}")
            status["model_loaded"] = False
            status["load_error"] = str(e)

    return status


@router.post("/train-model")
async def train_model(background_tasks: BackgroundTasks, dataset_path: str = None):
    from config.settings import Settings

    if dataset_path is None:
        dataset_path = os.path.join(Settings.DATA_DIR, 'cv_dataset.json')

    def train_background():
        logger.info(f"Iniciando entrenamiento en background con dataset: {dataset_path}")
        cv_app = get_cv_app()
        trainer = ModelTrainer(cv_app.classifier, cv_app.data_pipeline)
        success = trainer.train_from_json(dataset_path)
        logger.info(f"Entrenamiento completado: {'éxito' if success else 'fallido'}")

    background_tasks.add_task(train_background)

    return {
        "message": "Entrenamiento iniciado en segundo plano",
        "dataset_path": dataset_path,
        "started_at": datetime.now().isoformat()
    }