from fastapi import FastAPI
from api.app import create_fastapi_app
from api.deps import set_cv_app
from api.routes.main import router as main_router
from api.routes.analysis import router as analysis_router
from api.routes.model import router as model_router
from api.routes.skills import router as skills_router

def create_api_app(classifier_app) -> FastAPI:
    set_cv_app(classifier_app)
    app = create_fastapi_app()
    
    # Registrar rutas
    app.include_router(main_router)
    app.include_router(analysis_router, prefix="/api/v1")
    app.include_router(model_router, prefix="/api/v1")
    app.include_router(skills_router, prefix="/api/v1")
    
    return app

def start_server(classifier_app, host: str = "0.0.0.0", port: int = 8000):
    from uvicorn import Config, Server
    import logging
    
    app = create_api_app(classifier_app)
    display_host = "localhost" if host == "0.0.0.0" else host
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("🚀 INICIANDO CV CLASSIFIER API SERVER")
    logger.info("=" * 60)
    logger.info(f"📍 Servidor escuchando en: {host}:{port}")
    logger.info(f"🌐 Acceso local: http://127.0.0.1:{port}/")
    logger.info(f"📖 Documentación: http://127.0.0.1:{port}/docs")
    logger.info(f"❤️  Health check: http://127.0.0.1:{port}/health")
    if host == "0.0.0.0":
        logger.info(f"🔗 También disponible en: http://localhost:{port}/")
    logger.info("=" * 60)
    
    config = Config(app, host=host, port=port, log_level="info", access_log=True, reload=False, workers=1)
    server = Server(config)
    server.run()

def run_dev_server(classifier_app):
    start_server(classifier_app, host="127.0.0.1", port=8000)