"""
Módulo de Ingesta de Datos
===========================

Este módulo contiene funciones para cargar, validar y procesar los datos del call center.
Incluye funcionalidades para manejar diferentes formatos de entrada y validar la estructura.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, Any, Tuple
import logging
from pathlib import Path

# Configuramos el logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Clase para manejar la ingesta de datos del call center."""
    
    def __init__(self, data_path: str = "00_data/raw/"):
        """
        Inicializamos la clase de ingesta de datos.
        
        Args:
            data_path (str): Ruta a la carpeta de datos raw
        """
        self.data_path = Path(data_path)
        self.expected_columns = [
            'vru+line', 'call_id', 'customer_id', 'priority', 'type',
            'date', 'vru_entry', 'vru_exit', 'vru_time', 'q_start',
            'q_exit', 'q_time', 'outcome', 'ser_start', 'ser_exit',
            'ser_time', 'server'
        ]
    
    def load_call_center_data(self, filename: str = "Call_Center_1999_DataSet.csv") -> pd.DataFrame:
        """
        Cargamos el dataset principal del call center.
        
        Args:
            filename (str): Nombre del archivo CSV
            
        Returns:
            pd.DataFrame: Dataset cargado
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el archivo no tiene el formato esperado
        """
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        try:
            # Cargamos los datos con encoding adecuado
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
            
            # Validamos la estructura básica
            self._validate_basic_structure(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            raise
    
    def _validate_basic_structure(self, df: pd.DataFrame) -> None:
        """
        Validar la estructura básica del dataset.
        
        Args:
            df (pd.DataFrame): Dataset a validar
            
        Raises:
            ValueError: Si la estructura no es válida
        """
        # Verificamos el número mínimo de columnas
        if df.shape[1] < 15:
            raise ValueError(f"Dataset tiene muy pocas columnas: {df.shape[1]}")
        
        # Verificamos que no esté vacío
        if df.empty:
            raise ValueError("Dataset está vacío")
        
        logger.info("Validación básica de estructura completada")
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Obtener información básica del dataset.
        
        Args:
            df (pd.DataFrame): Dataset a analizar
            
        Returns:
            Dict[str, Any]: Información del dataset
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        return info
    
    def detect_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detectamos y sugerimos si los tipos de datos son los apropiados.
        
        Args:
            df (pd.DataFrame): Dataset a analizar
            
        Returns:
            Dict[str, str]: Mapeo de columnas a tipos sugeridos
        """
        type_suggestions = {}
        
        for col in df.columns:
            if 'id' in col.lower():
                type_suggestions[col] = 'Int64'  # Nullable integer
            elif 'time' in col.lower() and col != 'date':
                type_suggestions[col] = 'Int64'  # Tiempo en segundos
            elif col == 'date':
                type_suggestions[col] = 'datetime64[ns]'
            elif col in ['vru_entry', 'vru_exit', 'q_start', 'q_exit', 'ser_start', 'ser_exit']:
                type_suggestions[col] = 'object'  # Mantener como string para procesamiento
            elif col == 'priority':
                type_suggestions[col] = 'Int8'
            else:
                type_suggestions[col] = 'object'
        
        return type_suggestions


def load_data(filename: str = "Call_Center_1999_DataSet.csv", 
              data_path: str = "00_data/raw/") -> pd.DataFrame:
    """
    Función wrapper para cargar datos del call center.
    
    Args:
        filename (str): Nombre del archivo CSV
        data_path (str): Ruta a la carpeta de datos
        
    Returns:
        pd.DataFrame: Dataset cargado
    """
    ingestion = DataIngestion(data_path)
    return ingestion.load_call_center_data(filename)


def validate_data_format(df: pd.DataFrame) -> Tuple[bool, Dict[str, str]]:
    """
    Validamos  el formato de los datos cargados.
    
    Args:
        df (pd.DataFrame): Dataset a validar
        
    Returns:
        Tuple[bool, Dict[str, str]]: (es_válido, errores_encontrados)
    """
    errors = {}
    is_valid = True
    
    # Validaciones específicas del call center
    required_cols = ['call_id', 'customer_id', 'date', 'outcome']
    
    for col in required_cols:
        if col not in df.columns:
            errors[col] = f"Columna requerida '{col}' no encontrada"
            is_valid = False
    
    # Validamos que call_id sea único
    if 'call_id' in df.columns and df['call_id'].duplicated().any():
        errors['call_id'] = "call_id contiene valores duplicados"
        is_valid = False
    
    # Validamos los valores valores de outcome
    if 'outcome' in df.columns:
        valid_outcomes = ['AGENT', 'HANG', 'PHANTOM']
        invalid_outcomes = df[~df['outcome'].isin(valid_outcomes)]['outcome'].unique()
        if len(invalid_outcomes) > 0:
            errors['outcome'] = f"Valores inválidos en outcome: {invalid_outcomes}"
            is_valid = False
    
    return is_valid, errors


if __name__ == "__main__":
    # Ejemplo de uso
    try:
        df = load_data()
        print(f"Datos cargados: {df.shape}")
        print(f"Columnas: {list(df.columns)}")
        
        # Validamos el formato
        is_valid, errors = validate_data_format(df)
        if is_valid:
            print(" Datos válidos")
        else:
            print("❌ Errores encontrados:")
            for col, error in errors.items():
                print(f"  - {col}: {error}")
                
    except Exception as e:
        print(f"Error: {e}")
