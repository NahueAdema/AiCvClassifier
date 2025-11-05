import logging
from data.extractor import DocumentExtractor, CVInfoExtractor
from data.preprocessor import DataPipeline
from models.classifier import CVClassifier

logger = logging.getLogger(__name__)


class CVClassifierApp:
    def __init__(self):
        self.document_extractor = DocumentExtractor()
        self.cv_info_extractor = CVInfoExtractor()
        self.data_pipeline = DataPipeline()
        self.classifier = CVClassifier()
        self.job_matcher = None