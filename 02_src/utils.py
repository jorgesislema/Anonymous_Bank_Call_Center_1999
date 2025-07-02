"""
Módulo de Utilidades Generales
===============================

Este módulo contiene funciones de utilidad general para el proyecto de análisis
del call center, incluyendo configuración, logging y helpers varios.
"""

import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class Config:
    """Clase para manejar configuraciones del proyecto."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Inicializar configuración.
        
        Args:
            config_path (str): Ruta al archivo de configuración
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Cargar configuración desde archivo YAML."""
        default_config = {
            'data': {
                'raw_path': '00_data/raw/',
                'processed_path': '00_data/processed/',
                'external_path': '00_data/external/'
            },
            'analysis': {
                'outlier_threshold': 3.0,
                'missing_threshold': 0.1,
                'min_service_time': 10,
                'max_service_time': 3600
            },
            'visualization': {
                'figure_size': [12, 8],
                'dpi': 300,
                'style': 'seaborn-v0_8',
                'color_palette': 'viridis'
            },
            'modeling': {
                'test_size': 0.2,
                'random_state': 42,
                'cv_folds': 5
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    default_config.update(file_config)
            except Exception as e:
                logging.warning(f"Error al cargar configuración: {e}. Usando configuración por defecto.")
        
        return default_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtener valor de configuración con notación de punto."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


class ProjectLogger:
    """Clase para configurar logging del proyecto."""
    
    @staticmethod
    def setup_logging(log_level: str = "INFO", 
                     log_file: Optional[str] = None) -> logging.Logger:
        """
        Configurar logging para el proyecto.
        
        Args:
            log_level (str): Nivel de logging
            log_file (str, optional): Archivo de log
            
        Returns:
            logging.Logger: Logger configurado
        """
        # Crear directorio de logs si no existe
        if log_file:
            log_path = Path(log_file).parent
            log_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Configurar logger
        logger = logging.getLogger('call_center_analysis')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Handler para archivo si se especifica
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


class DataValidator:
    """Clase para validar datasets y datos."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, 
                          required_columns: List[str] = None,
                          min_rows: int = 1) -> Dict[str, Any]:
        """
        Validar un DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame a validar
            required_columns (List[str]): Columnas requeridas
            min_rows (int): Número mínimo de filas
            
        Returns:
            Dict[str, Any]: Resultado de validación
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum(),
                'null_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict()
            }
        }
        
        # Validar que no esté vacío
        if df.empty:
            validation_result['is_valid'] = False
            validation_result['errors'].append("DataFrame está vacío")
            return validation_result
        
        # Validar número mínimo de filas
        if len(df) < min_rows:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"DataFrame tiene menos de {min_rows} filas")
        
        # Validar columnas requeridas
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Columnas faltantes: {missing_cols}")
        
        # Advertencias por datos faltantes
        null_percentage = df.isnull().sum() / len(df)
        high_null_cols = null_percentage[null_percentage > 0.5].index.tolist()
        if high_null_cols:
            validation_result['warnings'].append(f"Columnas con >50% valores nulos: {high_null_cols}")
        
        return validation_result
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Verificar calidad general de los datos.
        
        Args:
            df (pd.DataFrame): DataFrame a verificar
            
        Returns:
            Dict[str, Any]: Reporte de calidad
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicate_rows': df.duplicated().sum(),
            'completely_null_rows': df.isnull().all(axis=1).sum(),
            'columns_with_nulls': df.isnull().any().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'string']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
        }
        
        # Calcular score de calidad (0-100)
        null_penalty = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 50
        duplicate_penalty = (quality_report['duplicate_rows'] / len(df)) * 30
        empty_row_penalty = (quality_report['completely_null_rows'] / len(df)) * 20
        
        quality_score = max(0, 100 - null_penalty - duplicate_penalty - empty_row_penalty)
        quality_report['quality_score'] = round(quality_score, 2)
        
        return quality_report


class FileManager:
    """Clase para manejar archivos del proyecto."""
    
    @staticmethod
    def ensure_directory(path: str) -> Path:
        """
        Asegurar que un directorio existe.
        
        Args:
            path (str): Ruta del directorio
            
        Returns:
            Path: Objeto Path del directorio
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, 
                      filepath: str, 
                      format: str = 'csv',
                      **kwargs) -> None:
        """
        Guardar DataFrame en diferentes formatos.
        
        Args:
            df (pd.DataFrame): DataFrame a guardar
            filepath (str): Ruta del archivo
            format (str): Formato ('csv', 'parquet', 'pickle', 'excel')
            **kwargs: Argumentos adicionales para el método de guardado
        """
        # Asegurar que el directorio existe
        FileManager.ensure_directory(Path(filepath).parent)
        
        if format.lower() == 'csv':
            df.to_csv(filepath, index=False, encoding='utf-8', **kwargs)
        elif format.lower() == 'parquet':
            df.to_parquet(filepath, index=False, **kwargs)
        elif format.lower() == 'pickle':
            df.to_pickle(filepath, **kwargs)
        elif format.lower() == 'excel':
            df.to_excel(filepath, index=False, **kwargs)
        else:
            raise ValueError(f"Formato no soportado: {format}")
    
    @staticmethod
    def load_dataframe(filepath: str, 
                      format: str = None,
                      **kwargs) -> pd.DataFrame:
        """
        Cargar DataFrame desde archivo.
        
        Args:
            filepath (str): Ruta del archivo
            format (str, optional): Formato específico
            **kwargs: Argumentos adicionales
            
        Returns:
            pd.DataFrame: DataFrame cargado
        """
        file_path = Path(filepath)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        # Detectar formato por extensión si no se especifica
        if format is None:
            format = file_path.suffix.lower().lstrip('.')
        
        if format == 'csv':
            return pd.read_csv(filepath, **kwargs)
        elif format == 'parquet':
            return pd.read_parquet(filepath, **kwargs)
        elif format == 'pickle':
            return pd.read_pickle(filepath, **kwargs)
        elif format in ['xlsx', 'xls']:
            return pd.read_excel(filepath, **kwargs)
        else:
            raise ValueError(f"Formato no soportado: {format}")


class DataProfiler:
    """Clase para generar perfiles de datos."""
    
    @staticmethod
    def generate_basic_profile(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generar perfil básico de datos.
        
        Args:
            df (pd.DataFrame): DataFrame a perfilar
            
        Returns:
            Dict[str, Any]: Perfil de datos
        """
        profile = {
            'overview': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            'columns': {}
        }
        
        for col in df.columns:
            col_profile = {
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'null_percentage': round(df[col].isnull().sum() / len(df) * 100, 2),
                'unique_count': int(df[col].nunique()),
                'unique_percentage': round(df[col].nunique() / len(df) * 100, 2)
            }
            
            # Estadísticas para columnas numéricas
            if df[col].dtype in ['int64', 'float64', 'Int64', 'Float64']:
                col_profile.update({
                    'mean': float(df[col].mean()) if df[col].notna().any() else None,
                    'std': float(df[col].std()) if df[col].notna().any() else None,
                    'min': float(df[col].min()) if df[col].notna().any() else None,
                    'max': float(df[col].max()) if df[col].notna().any() else None,
                    'median': float(df[col].median()) if df[col].notna().any() else None
                })
            
            # Valores más frecuentes para categóricas
            elif df[col].dtype in ['object', 'string']:
                value_counts = df[col].value_counts().head(5)
                col_profile['top_values'] = value_counts.to_dict()
            
            profile['columns'][col] = col_profile
        
        return profile


# Funciones de utilidad generales
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Wrapper para configurar logging."""
    return ProjectLogger.setup_logging(log_level, log_file)


def get_config(config_path: str = "config/config.yaml") -> Config:
    """Wrapper para obtener configuración."""
    return Config(config_path)


def create_project_structure() -> None:
    """Crear estructura de directorios del proyecto."""
    directories = [
        '00_data/raw',
        '00_data/processed', 
        '00_data/external',
        '01_notebooks/sandbox',
        '02_src',
        '03_dashboards/power_bi',
        '03_dashboards/streamlit',
        '04_reports/figures',
        'config',
        'tests',
        'logs'
    ]
    
    for directory in directories:
        FileManager.ensure_directory(directory)
    
    print("Estructura de proyecto creada exitosamente")


def format_time_duration(seconds: float) -> str:
    """
    Formatear duración en segundos a formato legible.
    
    Args:
        seconds (float): Duración en segundos
        
    Returns:
        str: Duración formateada
    """
    if pd.isna(seconds):
        return "N/A"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def calculate_business_hours_only(df: pd.DataFrame, 
                                 start_hour: int = 7, 
                                 end_hour: int = 24) -> pd.DataFrame:
    """
    Filtrar datos solo para horario comercial.
    
    Args:
        df (pd.DataFrame): Dataset
        start_hour (int): Hora de inicio
        end_hour (int): Hora de fin
        
    Returns:
        pd.DataFrame: Dataset filtrado
    """
    if 'vru_entry_hour' not in df.columns:
        return df
    
    return df[
        (df['vru_entry_hour'] >= start_hour) & 
        (df['vru_entry_hour'] < end_hour)
    ].copy()


if __name__ == "__main__":
    # Ejemplo de uso
    print("Utilidades del proyecto Call Center Analysis")
    
    # Configurar logging
    logger = setup_logging("INFO")
    logger.info("Logger configurado exitosamente")
    
    # Crear estructura de proyecto
    create_project_structure()
      # Mostrar configuración
    config = get_config()
    print(f"Ruta de datos raw: {config.get('data.raw_path')}")
    print(f"Tamaño de figura por defecto: {config.get('visualization.figure_size')}")
    
    print("Utilidades inicializadas correctamente")
