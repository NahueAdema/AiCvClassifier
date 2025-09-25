Acá te armé un **README.md** completo y bien explicado para tu proyecto de IA con TensorFlow que evalúa CVs técnicos de developers:

````markdown
# 📄 IA para Evaluación de CVs Técnicos de Developers

Este proyecto utiliza **TensorFlow** y **FastAPI** para analizar y evaluar CVs técnicos de desarrolladores, generando insights útiles para procesos de selección.  

El servicio expone una API REST accesible en `http://127.0.0.1:8000`, con documentación interactiva en **Swagger UI** (`/docs`).  

---

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/NahueAdema/AiCvClassifier
cd AiCvClassifier
````

### 2. Crear y activar entorno virtual

En **Linux/Mac**:

```bash
python3 -m venv venv
source venv/bin/activate
```

En **Windows (PowerShell)**:

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecución del servidor

Para iniciar la API en modo servidor:

```bash
python main.py --mode server
```

Por defecto, el servidor quedará levantado en:

```
http://127.0.0.1:8000
```

---

## 📚 Endpoints principales

* **Health Check**:
  `GET /health` → Verifica que el servicio está corriendo.

* **Subir CV para análisis**:
  `POST /analyze-cv` → Permite enviar un archivo `.pdf` o `.docx` para evaluación.

* **Swagger UI** (documentación interactiva):
  [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🛠 Tecnologías utilizadas

* [TensorFlow](https://www.tensorflow.org/) → Modelado y predicción.
* [FastAPI](https://fastapi.tiangolo.com/) → API REST.
* [Uvicorn](https://www.uvicorn.org/) → Servidor ASGI rápido para correr la API.
* [Python 3.9+](https://www.python.org/)

---

## 🤝 Contribución

1. Haz un fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Haz commit de tus cambios (`git commit -m 'Agregada nueva funcionalidad'`)
4. Haz push a tu rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request 🚀

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia MIT.

