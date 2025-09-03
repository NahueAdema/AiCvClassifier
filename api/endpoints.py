from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import tempfile
import os
import uvicorn
import logging
from config.settings import settings
 
logger = logging.getLogger(__name__)

# Modelos Pydantic para la API
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

# Variable global para la app de CV
cv_app = None

def create_api_app(classifier_app) -> FastAPI:
    """Crea la aplicación FastAPI"""
    global cv_app
    cv_app = classifier_app
    
    app = FastAPI(
        title="CV Classifier API",
        description="API para clasificación automática de currículums",
        version="1.0.0"
    )
    
    @app.get("/")
    async def root():
        return {"message": "CV Classifier API", "version": "1.0.0"}
    
    @app.get("/health")
    async def health_check():
        """Endpoint de salud"""
        return {"status": "healthy", "service": "cv-classifier"}
    
    @app.post("/analyze-cv", response_model=Dict)
    async def analyze_cv(
        file: UploadFile = File(...),
        job_requirements: Optional[str] = None
    ):
        """Analiza un CV individual"""
        try:
            # Validar tipo de archivo
            allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Tipo de archivo no soportado. Use: {', '.join(allowed_extensions)}"
                )
            
            # Guardar archivo temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                # Procesar requisitos del trabajo si se proporcionan
                job_req_dict = None
                if job_requirements:
                    import json
                    job_req_data = json.loads(job_requirements)
                    job_req_dict = cv_app.create_job_profile(
                        job_req_data.get('description', ''),
                        job_req_data.get('required_skills', []),
                        job_req_data.get('min_experience', 0),
                        job_req_data.get('preferred_education', 'bachelor')
                    )
                
                # Procesar CV
                result = cv_app.process_single_cv(tmp_file_path, job_req_dict)
                
                if 'error' in result:
                    raise HTTPException(status_code=500, detail=result['error'])
                
                # Limpiar datos sensibles antes de devolver
                clean_result = result.copy()
                clean_result.pop('file_path', None)
                
                return clean_result
                
            finally:
                # Limpiar archivo temporal
                os.unlink(tmp_file_path)
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error analizando CV: {e}")
            raise HTTPException(status_code=500, detail=f"Error procesando CV: {str(e)}")
    
    @app.post("/batch-analyze")
    async def batch_analyze_cvs(
        files: List[UploadFile] = File(...),
        background_tasks: BackgroundTasks = BackgroundTasks(),
        job_requirements: Optional[str] = None
    ):
        """Analiza múltiples CVs en lote"""
        try:
            if len(files) > 10:  # Límite para evitar sobrecarga
                raise HTTPException(
                    status_code=400, 
                    detail="Máximo 10 archivos por solicitud"
                )
            
            results = []
            temp_files = []
            
            # Procesar requisitos del trabajo
            job_req_dict = None
            if job_requirements:
                import json
                job_req_data = json.loads(job_requirements)
                job_req_dict = cv_app.create_job_profile(
                    job_req_data.get('description', ''),
                    job_req_data.get('required_skills', []),
                    job_req_data.get('min_experience', 0),
                    job_req_data.get('preferred_education', 'bachelor')
                )
            
            try:
                for file in files:
                    # Validar archivo
                    file_extension = os.path.splitext(file.filename)[1].lower()
                    allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
                    
                    if file_extension not in allowed_extensions:
                        results.append({
                            'filename': file.filename,
                            'error': f'Tipo de archivo no soportado: {file_extension}'
                        })
                        continue
                    
                    # Guardar temporalmente
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                        content = await file.read()
                        tmp_file.write(content)
                        temp_files.append(tmp_file.name)
                        
                        # Procesar CV
                        result = cv_app.process_single_cv(tmp_file.name, job_req_dict)
                        result['filename'] = file.filename
                        result.pop('file_path', None)  # Limpiar path temporal
                        
                        results.append(result)
                
                return {
                    'processed_count': len(results),
                    'results': results,
                    'job_requirements_applied': job_req_dict is not None
                }
                
            finally:
                # Limpiar archivos temporales
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
                        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error en análisis en lote: {e}")
            raise HTTPException(status_code=500, detail=f"Error en procesamiento: {str(e)}")
    
    @app.post("/create-job-profile")
    async def create_job_profile(job_req: JobRequirements):
        """Crea un perfil de trabajo para matching"""
        try:
            profile = cv_app.create_job_profile(
                job_req.description,
                job_req.required_skills,
                job_req.min_experience,
                job_req.preferred_education
            )
            return {"job_profile": profile, "status": "created"}
            
        except Exception as e:
            logger.error(f"Error creando perfil de trabajo: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/train-model")
    async def train_model(background_tasks: BackgroundTasks):
        """Entrena el modelo con datos de ejemplo (solo para desarrollo)"""
        try:
            # Ejecutar entrenamiento en background
            def train_background():
                success = cv_app.train_model_with_sample_data()
                logger.info(f"Entrenamiento completado: {'exitoso' if success else 'fallido'}")
            
            background_tasks.add_task(train_background)
            
            return {
                "message": "Entrenamiento iniciado en segundo plano",
                "status": "training_started"
            }
            
        except Exception as e:
            logger.error(f"Error iniciando entrenamiento: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/model-status")
    async def model_status():
        """Verifica el estado del modelo"""
        try:
            model_path = os.path.join(settings.MODEL_DIR, 'cv_classifier_model')
            model_exists = os.path.exists(model_path)
            
            status = {
                "model_trained": model_exists,
                "model_path": model_path if model_exists else None,
                "classifier_ready": cv_app.classifier.model is not None
            }
            
            if model_exists:
                try:
                    # Intentar cargar modelo si no está cargado
                    if not cv_app.classifier.model:
                        cv_app.classifier.load_model(model_path)
                    status["model_loaded"] = True
                except:
                    status["model_loaded"] = False
            
            return status
            
        except Exception as e:
            logger.error(f"Error verificando estado del modelo: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/skills-available")
    async def get_available_skills():
        """Devuelve las habilidades disponibles para filtrado"""
        return {
            "tech_skills": settings.TECH_SKILLS,
            "soft_skills": settings.SOFT_SKILLS,
            "all_skills": settings.get_all_skills()
        }
    
    return app

def start_server(classifier_app):
    """Inicia el servidor de la API"""
    app = create_api_app(classifier_app)
    
    logger.info(f"Iniciando servidor en {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(
        app, 
        host=settings.API_HOST, 
        port=settings.API_PORT,
        log_level="info"
    )