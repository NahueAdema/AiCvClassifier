# main.py
import os
import sys
import logging
import argparse
import json
from nltk_setup import download_nltk_resources

download_nltk_resources()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar nuevos módulos
from core.app import CVClassifierApp
from core.processor import CVProcessor
from core.trainer import ModelTrainer
from core.validator import CVValidator


def main():
    parser = argparse.ArgumentParser(description='Clasificador de CVs con IA - Con filtro de perfiles técnicos')
    parser.add_argument('--mode', choices=['train', 'process', 'batch', 'validate', 'server'],
                       default='process', help='Modo de operación')
    parser.add_argument('--input', help='Archivo CV o carpeta para procesar')
    parser.add_argument('--output', help='Archivo de salida para resultados')
    parser.add_argument('--train-data', help='Ruta a datos de entrenamiento')
    parser.add_argument('--text', help='Texto del CV para validación rápida')

    args = parser.parse_args()

    app = CVClassifierApp()
    processor = CVProcessor(app)
    trainer = ModelTrainer(app.classifier, app.data_pipeline)

    if args.mode == 'train':
        logger.info("Modo entrenamiento")
        success = trainer.train_from_json(args.train_data)
        if success:
            print("✅ Modelo entrenado exitosamente")
        else:
            print("❌ Error en el entrenamiento")

    elif args.mode == 'validate':
        if args.text:
            result = CVValidator.quick_validate(args.text)
        elif args.input:
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
            result = CVValidator.quick_validate(text)
        else:
            print("❌ Para validación usa --text 'texto del CV' o --input archivo.txt")
            return

        print(f"🔍 VALIDACIÓN RÁPIDA:")
        print(f"   Resultado: {result['validation_result']}")
        print(f"   Score técnico: {result['tech_score']:.2f}")
        print(f"   Recomendación: {result['recommendation']}")
        summary = result.get('detailed_assessment', {}).get('executive_summary', '')
        print(f"   Análisis Detallado: {summary}")

    elif args.mode == 'process':
        if not args.input:
            print("❌ Debes especificar un archivo con --input")
            return
        logger.info(f"Procesando CV: {args.input}")
        result = processor.process_single_cv(args.input)
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ CV procesado exitosamente:")
            print(f"   📄 Archivo: {os.path.basename(result['file_path'])}")
            print(f"   🎯 Clasificación: {result['predicted_class']}")
            print(f"   📊 Score: {result.get('cv_score', 0):.1f}/100")
            print(f"   🔧 Score técnico: {result.get('tech_score', 0):.2f}")
            print(f"   ⚡ Etapa: {result.get('validation_stage', 'unknown')}")
            if 'detailed_analysis' in result:
                analysis = result['detailed_analysis']
                print(f"   📋 Resumen: {analysis.get('executive_summary', '')}")
                top_skills = analysis.get('strengths_analysis', {}).get('technical_skills', {}).get('top_skills', [])
                print(f"   💪 Fortalezas: {len(top_skills)} habilidades técnicas")
            if result.get('rejection_reason'):
                print(f"   ❌ Razón rechazo: {result['rejection_reason']}")

    elif args.mode == 'batch':
        if not args.input:
            print("❌ Debes especificar una carpeta con --input")
            return
        logger.info(f"Procesamiento en lote: {args.input}")
        results = processor.batch_process_cvs(args.input, args.output)
        print(f"✅ Procesados {len(results)} CVs")
        apto = sum(1 for r in results if r.get('predicted_class') == 'Apto')
        revisar = sum(1 for r in results if r.get('predicted_class') == 'Revisar')
        no_apto = sum(1 for r in results if r.get('predicted_class') == 'No apto')
        errores = sum(1 for r in results if 'error' in r)
        print(f"📊 RESUMEN:")
        print(f"   ✅ Aptos: {apto}")
        print(f"   🔍 Revisar: {revisar}")
        print(f"   ❌ No aptos: {no_apto}")
        print(f"   ⚠️  Errores: {errores}")

    elif args.mode == 'server':
        print("🚀 Iniciando servidor API...")
        from api.endpoints import start_server
        start_server(app)


if __name__ == "__main__":
    main()