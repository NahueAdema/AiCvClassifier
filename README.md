**Entrenar modelo**
python main.py --mode train

**Procesar un CV**
python main.py --mode process --input **"cv_ejemplo.pdf"**

cv_supermercado

**Iniciar API**
python main.py --mode server

pip **install** -r requirements.txt

# Validación rápida de texto

python main.py --mode validate --text "Soy cajero de supermercado con 3 años de experiencia"

# Validación de archivo

python main.py --mode validate --input cv_sospechoso.txt

# Procesamiento normal (mejorado)

python main.py --mode process --input cv.pdf

# Batch con estadísticas detalladas

python main.py --mode batch --input carpeta_cvs/ --output resultados.json
