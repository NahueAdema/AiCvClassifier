from fastapi import Depends
from typing import Any

# Variable global que será inyectada desde create_api_app
_cv_app_instance = None

def set_cv_app(app_instance: Any):
    global _cv_app_instance
    _cv_app_instance = app_instance

def get_cv_app():
    if _cv_app_instance is None:
        raise RuntimeError("CV App no inicializada")
    return _cv_app_instance