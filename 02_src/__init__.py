"""
Call Center Analysis Package

Este paquete contiene módulos para análisis completo de datos del call center.
Incluye funcionalidades para ingesta, limpieza, análisis y visualización de datos.

Módulos:
    - data_ingestion: Funciones para cargar y importar datos
    - data_cleaning: Limpieza y validación de datos
    - feature_engineering: Creación de variables derivadas
    - analysis: Funciones de análisis estadístico y descriptivo
    - modeling: Modelos de machine learning
    - visualization: Gráficos y visualizaciones
    - utils: Utilidades generales

Autor: Equipo de Data Science
Fecha: Mayo 2025
"""

__version__ = "1.0.0"
__author__ = "Equipo de Data Science"
__email__ = "data-team@anonymousbank.com"

# Importaciones principales
from .data_ingestion import load_data, validate_data_format
from .data_cleaning import clean_dataset, remove_outliers
from .feature_engineering import create_features, engineer_time_features
from .utils import setup_logging, get_config
