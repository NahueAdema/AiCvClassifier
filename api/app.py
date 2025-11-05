from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

def create_fastapi_app() -> FastAPI:
    app = FastAPI(
        title="CV Classifier API",
        description="API para clasificación automática de currículums con validación técnica previa",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    static_path = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_path):
        app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    return app