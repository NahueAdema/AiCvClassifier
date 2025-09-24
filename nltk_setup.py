# nltk_setup.py
import nltk

def download_nltk_resources():
    """Descarga todos los recursos necesarios de NLTK"""
    resources = [
        'punkt',
        'punkt_tab', 
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger'
    ]
    
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
            print(f"✓ {resource} ya está instalado")
        except LookupError:
            print(f"📥 Descargando {resource}...")
            nltk.download(resource)
            print(f"✓ {resource} descargado correctamente")

if __name__ == "__main__":
    download_nltk_resources()