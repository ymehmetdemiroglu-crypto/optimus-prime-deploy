# Competitive Intelligence Module
from .models import *
from .service import CompetitiveIntelligenceService
from .detectors import PriceChangeDetector, CannibalizationDetector
from .forecasting import Forecaster
from .strategy import UndercutPredictor, GameTheorySimulator
