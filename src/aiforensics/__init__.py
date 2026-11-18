"""Image Detector - AI image forensics toolkit.

Detect, localize and attribute AI-generated images.
"""

from aiforensics.config import settings
from aiforensics.schemas import AnalysisResult

__all__ = ["AnalysisResult", "settings", "__version__"]

__version__ = settings.version
