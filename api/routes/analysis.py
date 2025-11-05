from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from api.models import ValidateTextRequest
from api.deps import get_cv_app
import tempfile
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/validate-quick")
async def validate_cv_text(request: ValidateTextRequest):
    try:
        cv_app = get_cv_app()
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

@router.post("/analyze-cv")
async def analyze_cv(
    file: UploadFile = File(...),
    job_requirements: str = Form(None)
):
    try:
        cv_app = get_cv_app()
        allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Tipo de archivo no soportado. Use: {', '.join(allowed_extensions)}")

        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Archivo demasiado grande. Máximo 10MB permitido.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
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

            result = cv_app.process_single_cv(tmp_file_path, job_req_dict)
            if 'error' in result:
                raise HTTPException(status_code=500, detail=result['error'])

            clean_result = result.copy()
            clean_result.pop('file_path', None)
            clean_result['filename'] = file.filename
            clean_result['file_size_kb'] = round(len(content) / 1024, 2)
            clean_result['processed_at'] = datetime.now().isoformat()
            return clean_result

        finally:
            try:
                os.unlink(tmp_file_path)
            except:
                pass

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analizando CV: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando CV: {str(e)}")

@router.post("/batch-analyze")
async def batch_analyze_cvs(
    files: list[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    job_requirements: str = Form(None)
):
    from api.deps import get_cv_app
    cv_app = get_cv_app()

    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Máximo 20 archivos por solicitud")

    results = []
    temp_files = []
    stats = {'total': len(files), 'apto': 0, 'revisar': 0, 'no_apto': 0, 'errores': 0, 'rechazados_prevalidacion': 0}
    start_time = datetime.now()

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
        for file in files:
            file_extension = os.path.splitext(file.filename)[1].lower()
            allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
            if file_extension not in allowed_extensions:
                results.append({'filename': file.filename, 'error': f'Tipo no soportado: {file_extension}'})
                stats['errores'] += 1
                continue

            content = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(content)
                temp_files.append(tmp_file.name)
                result = cv_app.process_single_cv(tmp_file.name, job_req_dict)
                result.update({
                    'filename': file.filename,
                    'file_size_kb': round(len(content) / 1024, 2),
                    'processed_at': datetime.now().isoformat()
                })
                result.pop('file_path', None)

                if 'error' not in result:
                    cls = result.get('predicted_class', 'Error')
                    if cls == 'Apto': stats['apto'] += 1
                    elif cls == 'Revisar': stats['revisar'] += 1
                    elif cls == 'No apto':
                        stats['no_apto'] += 1
                        if result.get('validation_stage') == 'pre_validation':
                            stats['rechazados_prevalidacion'] += 1
                else:
                    stats['errores'] += 1

                results.append(result)

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
                'avg_time_per_cv': round(stats['processing_time'] / len(files), 2) if files else 0
            }
        }

    finally:
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass