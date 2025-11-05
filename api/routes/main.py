from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def root():
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
    return HTMLResponse(content=html_content)

@router.get("/health")
async def health_check():
    from config.settings import Settings
    import os
    model_path = os.path.join(Settings.MODEL_DIR, 'cv_classifier_model')
    model_exists = os.path.exists(model_path)

    # Intentar obtener app desde deps si está lista
    try:
        from api.deps import get_cv_app
        classifier_loaded = get_cv_app().classifier.model is not None
    except:
        classifier_loaded = False

    return {
        "status": "healthy",
        "service": "cv-classifier",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "model_available": model_exists,
        "classifier_loaded": classifier_loaded
    }

@router.get("/stats")
async def get_system_stats():
    try:
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        from api.deps import get_cv_app
        model_ready = get_cv_app().classifier.model is not None

        return {
            "system": {
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            },
            "service": {
                "version": "2.0.0",
                "model_ready": model_ready
            },
            "timestamp": datetime.now().isoformat()
        }
    except ImportError:
        from api.deps import get_cv_app
        model_ready = get_cv_app().classifier.model is not None
        return {
            "system": {"info": "System stats not available (psutil not installed)"},
            "service": {"version": "2.0.0", "model_ready": model_ready},
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error en /stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))