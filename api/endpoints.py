from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import tempfile
import os
import uvicorn
import logging
import json
import asyncio
from datetime import datetime
import shutil

# Configurar logging
logging.basicConfig(level=logging.INFO)
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

class ValidateTextRequest(BaseModel):
    text: str

# Variable global para la app de CV
cv_app = None

def create_api_app(classifier_app) -> FastAPI:
    """Crea la aplicación FastAPI con todas las funcionalidades"""
    global cv_app
    cv_app = classifier_app
    
    app = FastAPI(
        title="CV Classifier API",
        description="API para clasificación automática de currículums con validación técnica previa",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configurar CORS para permitir acceso desde el frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # En producción, especificar dominios exactos
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Servir archivos estáticos si existe carpeta 'static'
    static_path = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_path):
        app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Página principal con información de la API"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CV Classifier API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .method { color: white; padding: 4px 8px; border-radius: 3px; font-weight: bold; }
                .get { background: #27ae60; }
                .post { background: #3498db; }
                code { background: #2c3e50; color: white; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="header">🤖 CV Classifier API v2.0</h1>
                <p>API para clasificación automática de currículums con validación técnica.</p>
                
                <h2>📋 Endpoints Disponibles:</h2>
                
                <div class="endpoint">
                    <span class="method get">GET</span> <code>/health</code>
                    <p>Verificar estado del servicio</p>
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span> <code>/analyze-cv</code>
                    <p>Analizar un CV individual (archivo + requisitos opcionales)</p>
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span> <code>/validate-quick</code>
                    <p>Validación rápida de texto de CV sin procesar archivo</p>
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span> <code>/batch-analyze</code>
                    <p>Análisis en lote de múltiples CVs</p>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span> <code>/model-status</code>
                    <p>Estado del modelo de clasificación</p>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span> <code>/skills-available</code>
                    <p>Lista de habilidades técnicas disponibles</p>
                </div>
                
                <h2>📚 Documentación:</h2>
                <p>
                    <a href="/docs" target="_blank">📖 Swagger UI</a> | 
                    <a href="/redoc" target="_blank">📘 ReDoc</a>
                </p>
                
                <h2>🔧 Estado del Sistema:</h2>
                <p id="status">Verificando...</p>
            </div>
            
            <script>
                fetch('/health')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').innerHTML = 
                            `✅ ${data.status} - Servicio activo desde ${new Date().toLocaleString()}`;
                    })
                    .catch(() => {
                        document.getElementById('status').innerHTML = '❌ Error conectando al servicio';
                    });
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)
    
    @app.get("/health")
    async def health_check():
        """Endpoint de salud con información del sistema"""
        try:
            # Verificar estado del modelo
            from config.settings import Settings
            model_path = os.path.join(Settings.MODEL_DIR, 'cv_classifier_model')
            model_exists = os.path.exists(model_path)
            
            return {
                "status": "healthy",
                "service": "cv-classifier",
                "version": "2.0.0",
                "timestamp": datetime.now().isoformat(),
                "model_available": model_exists,
                "classifier_loaded": cv_app.classifier.model is not None if cv_app else False
            }
        except Exception as e:
            logger.error(f"Error en health check: {e}")
            raise HTTPException(status_code=500, detail="Health check failed")
    
    @app.post("/validate-quick")
    async def validate_cv_text(request: ValidateTextRequest):
        """Validación rápida de texto de CV sin necesidad de archivo"""
        try:
            result = cv_app.validate_cv_quick(request.text)
            
            return {
                "validation_result": result['validation_result'],
                "tech_score": result['tech_score'],
                "recommendation": result['recommendation'],
                "debug_info": result['debug_analysis'],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en validación rápida: {e}")
            raise HTTPException(status_code=500, detail=f"Error validando texto: {str(e)}")
    
    @app.post("/analyze-cv")
    async def analyze_cv(
        file: UploadFile = File(...),
        job_requirements: Optional[str] = Form(None)
    ):
        """Analiza un CV individual con validación técnica previa"""
        try:
            # Validar tipo de archivo
            allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Tipo de archivo no soportado. Use: {', '.join(allowed_extensions)}"
                )
            
            # Validar tamaño del archivo (máximo 10MB)
            file_size = 0
            content = await file.read()
            file_size = len(content)
            
            if file_size > 10 * 1024 * 1024:  # 10MB
                raise HTTPException(
                    status_code=400,
                    detail="Archivo demasiado grande. Máximo 10MB permitido."
                )
            
            # Guardar archivo temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                # Procesar requisitos del trabajo si se proporcionan
                job_req_dict = None
                if job_requirements:
                    try:
                        job_req_data = json.loads(job_requirements)
                        job_req_dict = cv_app.create_job_profile(
                            job_req_data.get('description', ''),
                            job_req_data.get('required_skills', []),
                            job_req_data.get('min_experience', 0),
                            job_req_data.get('preferred_education', 'bachelor')
                        )
                    except json.JSONDecodeError:
                        logger.warning("Formato JSON inválido en job_requirements")
                
                # Procesar CV
                result = cv_app.process_single_cv(tmp_file_path, job_req_dict)
                
                if 'error' in result:
                    raise HTTPException(status_code=500, detail=result['error'])
                
                # Limpiar datos sensibles y añadir metadata
                clean_result = result.copy()
                clean_result.pop('file_path', None)
                clean_result['filename'] = file.filename
                clean_result['file_size_kb'] = round(file_size / 1024, 2)
                clean_result['processed_at'] = datetime.now().isoformat()
                
                return clean_result
                
            finally:
                # Limpiar archivo temporal
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error analizando CV: {e}")
            raise HTTPException(status_code=500, detail=f"Error procesando CV: {str(e)}")
    
    @app.post("/batch-analyze")
    async def batch_analyze_cvs(
        files: List[UploadFile] = File(...),
        background_tasks: BackgroundTasks = BackgroundTasks(),
        job_requirements: Optional[str] = Form(None)
    ):
        """Analiza múltiples CVs en lote con estadísticas detalladas"""
        try:
            if len(files) > 20:  # Aumentado el límite
                raise HTTPException(
                    status_code=400, 
                    detail="Máximo 20 archivos por solicitud"
                )
            
            results = []
            temp_files = []
            stats = {
                'total': len(files),
                'apto': 0,
                'revisar': 0,
                'no_apto': 0,
                'errores': 0,
                'rechazados_prevalidacion': 0,
                'processing_time': 0
            }
            
            start_time = datetime.now()
            
            # Procesar requisitos del trabajo
            job_req_dict = None
            if job_requirements:
                try:
                    job_req_data = json.loads(job_requirements)
                    job_req_dict = cv_app.create_job_profile(
                        job_req_data.get('description', ''),
                        job_req_data.get('required_skills', []),
                        job_req_data.get('min_experience', 0),
                        job_req_data.get('preferred_education', 'bachelor')
                    )
                except json.JSONDecodeError:
                    logger.warning("Formato JSON inválido en job_requirements")
            
            try:
                for i, file in enumerate(files, 1):
                    logger.info(f"Procesando archivo {i}/{len(files)}: {file.filename}")
                    
                    # Validar archivo
                    file_extension = os.path.splitext(file.filename)[1].lower()
                    allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
                    
                    if file_extension not in allowed_extensions:
                        results.append({
                            'filename': file.filename,
                            'error': f'Tipo de archivo no soportado: {file_extension}',
                            'processed_at': datetime.now().isoformat()
                        })
                        stats['errores'] += 1
                        continue
                    
                    # Guardar temporalmente
                    try:
                        content = await file.read()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                            tmp_file.write(content)
                            temp_files.append(tmp_file.name)
                            
                            # Procesar CV
                            result = cv_app.process_single_cv(tmp_file.name, job_req_dict)
                            result['filename'] = file.filename
                            result['file_size_kb'] = round(len(content) / 1024, 2)
                            result['processed_at'] = datetime.now().isoformat()
                            result.pop('file_path', None)  # Limpiar path temporal
                            
                            # Actualizar estadísticas
                            if 'error' not in result:
                                classification = result.get('predicted_class', 'Error')
                                if classification == 'Apto':
                                    stats['apto'] += 1
                                elif classification == 'Revisar':
                                    stats['revisar'] += 1
                                elif classification == 'No apto':
                                    stats['no_apto'] += 1
                                    if result.get('validation_stage') == 'pre_validation':
                                        stats['rechazados_prevalidacion'] += 1
                            else:
                                stats['errores'] += 1
                            
                            results.append(result)
                            
                    except Exception as file_error:
                        logger.error(f"Error procesando {file.filename}: {file_error}")
                        results.append({
                            'filename': file.filename,
                            'error': str(file_error),
                            'processed_at': datetime.now().isoformat()
                        })
                        stats['errores'] += 1
                
                # Calcular tiempo de procesamiento
                end_time = datetime.now()
                stats['processing_time'] = (end_time - start_time).total_seconds()
                
                return {
                    'processed_count': len(results),
                    'statistics': stats,
                    'results': results,
                    'job_requirements_applied': job_req_dict is not None,
                    'processing_info': {
                        'start_time': start_time.isoformat(),
                        'end_time': end_time.isoformat(),
                        'avg_time_per_cv': round(stats['processing_time'] / len(files), 2) if len(files) > 0 else 0
                    }
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
            return {
                "job_profile": profile, 
                "status": "created",
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creando perfil de trabajo: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/train-model")
    async def train_model(background_tasks: BackgroundTasks):
        """Entrena el modelo con datos de ejemplo (desarrollo/testing)"""
        try:
            # Ejecutar entrenamiento en background
            def train_background():
                logger.info("Iniciando entrenamiento en background...")
                success = cv_app.train_model_with_sample_data()
                status = "exitoso" if success else "fallido"
                logger.info(f"Entrenamiento completado: {status}")
                return success
            
            background_tasks.add_task(train_background)
            
            return {
                "message": "Entrenamiento iniciado en segundo plano",
                "status": "training_started",
                "started_at": datetime.now().isoformat(),
                "note": "El modelo se entrenará con datos de ejemplo. Verificar /model-status para el progreso."
            }
            
        except Exception as e:
            logger.error(f"Error iniciando entrenamiento: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/model-status")
    async def model_status():
        """Verifica el estado completo del modelo y sistema"""
        try:
            from config.settings import Settings
            model_path = os.path.join(Settings.MODEL_DIR, 'cv_classifier_model')
            model_exists = os.path.exists(model_path)
            
            status = {
                "model_trained": model_exists,
                "model_path": model_path if model_exists else None,
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
                    "non_tech_sectors_count": len(Settings.NON_TECH_SECTORS)
                },
                "checked_at": datetime.now().isoformat()
            }
            
            if model_exists:
                try:
                    # Obtener información del modelo
                    model_stat = os.stat(model_path)
                    status["model_info"] = {
                        "created_at": datetime.fromtimestamp(model_stat.st_ctime).isoformat(),
                        "modified_at": datetime.fromtimestamp(model_stat.st_mtime).isoformat(),
                        "size_mb": round(model_stat.st_size / (1024 * 1024), 2)
                    }
                    
                    # Intentar cargar modelo si no está cargado
                    if not cv_app.classifier.model:
                        cv_app.classifier.load_model(model_path)
                    status["model_loaded"] = cv_app.classifier.model is not None
                    
                except Exception as load_error:
                    logger.warning(f"Error cargando modelo: {load_error}")
                    status["model_loaded"] = False
                    status["load_error"] = str(load_error)
            
            return status
            
        except Exception as e:
            logger.error(f"Error verificando estado del modelo: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/skills-available")
    async def get_available_skills():
        """Devuelve las habilidades y configuraciones disponibles"""
        try:
            from config.settings import Settings
            
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
            logger.error(f"Error obteniendo skills disponibles: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Endpoint adicional para estadísticas del sistema
    @app.get("/stats")
    async def get_system_stats():
        """Estadísticas del sistema y uso"""
        try:
            import psutil
            
            # Información del sistema
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "system": {
                    "memory_percent": memory.percent,
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "disk_percent": disk.percent,
                    "disk_free_gb": round(disk.free / (1024**3), 2)
                },
                "service": {
                    "uptime": "Service started",  # Podrías implementar un contador real
                    "version": "2.0.0",
                    "model_ready": cv_app.classifier.model is not None
                },
                "timestamp": datetime.now().isoformat()
            }
        except ImportError:
            # Si psutil no está disponible
            return {
                "system": {
                    "info": "System stats not available (psutil not installed)"
                },
                "service": {
                    "version": "2.0.0",
                    "model_ready": cv_app.classifier.model is not None
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

def start_server(classifier_app, host: str = "0.0.0.0", port: int = 8000):
    """Inicia el servidor de la API con configuración optimizada"""
    app = create_api_app(classifier_app)
    
    # Determinar URL accesible para mostrar al usuario
    display_host = "localhost" if host == "0.0.0.0" else host
    
    logger.info("=" * 60)
    logger.info("🚀 INICIANDO CV CLASSIFIER API SERVER")
    logger.info("=" * 60)
    logger.info(f"📍 Servidor escuchando en: {host}:{port}")
    logger.info(f"🌐 Acceso local: http://127.0.0.1:{port}/")
    logger.info(f"📖 Documentación: http://127.0.0.1:{port}/docs")
    logger.info(f"🏠 Página principal: http://127.0.0.1:{port}/")
    logger.info(f"❤️  Health check: http://127.0.0.1:{port}/health")
    if host == "0.0.0.0":
        logger.info(f"🔗 También disponible en: http://localhost:{port}/")
    logger.info("=" * 60)
    
    # Configuración del servidor
    config = uvicorn.Config(
        app, 
        host=host, 
        port=port,
        log_level="info",
        access_log=True,
        reload=False,  # Cambiar a True solo en desarrollo
        workers=1  # Para evitar problemas con el modelo compartido
    )
    
    server = uvicorn.Server(config)
    server.run()

# Función de conveniencia para desarrollo
def run_dev_server(classifier_app):
    """Ejecuta servidor en modo desarrollo"""
    start_server(classifier_app, host="127.0.0.1", port=8000)