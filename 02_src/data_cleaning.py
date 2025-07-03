"""
Módulo de Limpieza de Datos
============================

En este módulo, creamos funciones para limpiar, validar y preparar los datos del call center
para análisis posterior. Incluye manejo de valores faltantes, outliers y inconsistencias.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import logging
from datetime import datetime, time
import re

# Configuramos el logger para registrar el proceso de limpieza
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Clase para limpiar y validar datos del call center."""
    
    def __init__(self):
        """Inicializamos el limpiador de datos."""
        self.cleaning_report = {}
        
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realizamos el proceso completo de limpieza del dataset.
        
        Args:
            df (pd.DataFrame): Dataset crudo
            
        Returns:
            pd.DataFrame: Dataset limpio
        """
        logger.info("Iniciamos el proceso de limpieza de datos")
        
        df_clean = df.copy()
        
        # 1. Limpiamos los nombres de las columnas
        df_clean = self._clean_column_names(df_clean)
        
        # 2. Convertimos los tipos de los datos
        df_clean = self._convert_data_types(df_clean)
        
        # 3. Limpiamos los valores faltantes
        df_clean = self._handle_missing_values(df_clean)
        
        # 4. Validamos y limpiamos datos de tiempo
        df_clean = self._clean_time_data(df_clean)
        
        # 5. Validamos los datos del negocio
        df_clean = self._validate_business_rules(df_clean)
        
        # 6. Removemos outliers
        df_clean = self._remove_outliers(df_clean)
        
        # 7. Creamos variables derivadas básicas
        df_clean = self._create_derived_variables(df_clean)
        
        logger.info(f"Limpieza completada. Filas: {len(df)} -> {len(df_clean)}")
        
        return df_clean
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpiamos nombres de columnas."""
        df_clean = df.copy()
        
        # Creamos un mapeo de nombres de columnas
        column_mapping = {
            'vru+line': 'vru_line',
            'ser_start': 'service_start',
            'ser_exit': 'service_exit', 
            'ser_time': 'service_time'
        }
        
        df_clean = df_clean.rename(columns=column_mapping)
        
        # Limpiamos espacios y caracteres especiales
        df_clean.columns = df_clean.columns.str.strip().str.lower()
        
        self.cleaning_report['column_mapping'] = column_mapping
        
        return df_clean
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convertimos tipos de datos apropiados."""
        df_clean = df.copy()
        
        type_conversions = {}
        
        try:
            # Convertimos IDs a enteros nullable
            if 'call_id' in df_clean.columns:
                df_clean['call_id'] = pd.to_numeric(df_clean['call_id'], errors='coerce').astype('Int64')
                type_conversions['call_id'] = 'Int64'
            
            if 'customer_id' in df_clean.columns:
                df_clean['customer_id'] = pd.to_numeric(df_clean['customer_id'], errors='coerce').astype('Int64')
                type_conversions['customer_id'] = 'Int64'
            
            # Convertimos priority a entero pequeño
            if 'priority' in df_clean.columns:
                df_clean['priority'] = pd.to_numeric(df_clean['priority'], errors='coerce').astype('Int8')
                type_conversions['priority'] = 'Int8'
            
            # Convertimos tiempos a enteros
            time_columns = ['vru_time', 'q_time', 'service_time']
            for col in time_columns:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype('Int64')
                    type_conversions[col] = 'Int64'
            
            # Parseamos la fecha
            if 'date' in df_clean.columns:
                df_clean['date'] = self._parse_date_column(df_clean['date'])
                type_conversions['date'] = 'datetime64[ns]'
            
            # Convertimos campos de texto a string
            text_columns = ['type', 'outcome', 'server', 'vru_line']
            for col in text_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].astype('string')
                    type_conversions[col] = 'string'
                    
        except Exception as e:
            logger.error(f"Error en conversión de tipos: {e}")
        
        self.cleaning_report['type_conversions'] = type_conversions
        
        return df_clean
    
    def _parse_date_column(self, date_series: pd.Series) -> pd.Series:
        """Parsear columna de fecha en formato YYMMDD."""
        def parse_date(date_str):
            try:
                if pd.isna(date_str):
                    return pd.NaT
                
                date_str = str(date_str).strip()
                
                # Formato YYMMDD
                if len(date_str) == 6:
                    year = int('19' + date_str[:2])  # Asumimos 1999
                    month = int(date_str[2:4])
                    day = int(date_str[4:6])
                    return datetime(year, month, day)
                
                return pd.NaT
                
            except:
                return pd.NaT
        
        return date_series.apply(parse_date)
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manejar valores faltantes según reglas de negocio."""
        df_clean = df.copy()
        missing_info = {}
        
        # Para customer_id = 0, convertimos  a NaN 
        if 'customer_id' in df_clean.columns:
            zero_customers = (df_clean['customer_id'] == 0).sum()
            df_clean.loc[df_clean['customer_id'] == 0, 'customer_id'] = pd.NA
            missing_info['customer_id_zeros_converted'] = zero_customers
        
        # Tiempos de cola y servicio dependen del outcome
        if 'outcome' in df_clean.columns:
            # Si outcome != 'AGENT', service_time debería ser NaN
            mask_no_agent = df_clean['outcome'] != 'AGENT'
            if 'service_time' in df_clean.columns:
                df_clean.loc[mask_no_agent, 'service_time'] = pd.NA
            
            # Si outcome == 'HANG', puede que no haya tiempos de servicio
            mask_hang = df_clean['outcome'] == 'HANG'
            service_cols = ['service_start', 'service_exit', 'service_time']
            for col in service_cols:
                if col in df_clean.columns:
                    df_clean.loc[mask_hang, col] = pd.NA
        
        self.cleaning_report['missing_values'] = missing_info
        
        return df_clean
    
    def _clean_time_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpiar y validar datos de tiempo."""
        df_clean = df.copy()
        time_issues = {}
        
        # Validamos que los tiempos sean positivos
        time_columns = ['vru_time', 'q_time', 'service_time']
        for col in time_columns:
            if col in df_clean.columns:
                negative_count = (df_clean[col] < 0).sum()
                if negative_count > 0:
                    time_issues[f'{col}_negative'] = negative_count
                    df_clean.loc[df_clean[col] < 0, col] = pd.NA
        
        # Validamos la consistencia: vru_time + q_time + service_time debería ser razonable
        if all(col in df_clean.columns for col in time_columns):
            total_time = df_clean[time_columns].sum(axis=1, skipna=True)
            
            # Identificamos las  llamadas con tiempos extremos (más de 2 horas = 7200 segundos)
            extreme_times = total_time > 7200
            time_issues['extreme_total_time'] = extreme_times.sum()
        
        self.cleaning_report['time_data_issues'] = time_issues
        
        return df_clean
    
    def _validate_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validamos las  reglas de negocio específicas del call center."""
        df_clean = df.copy()
        validation_issues = {}
        
        # Validamos valores de priority
        if 'priority' in df_clean.columns:
            valid_priorities = [0, 1, 2]
            invalid_priority = ~df_clean['priority'].isin(valid_priorities)
            validation_issues['invalid_priority'] = invalid_priority.sum()
            df_clean.loc[invalid_priority, 'priority'] = pd.NA
        
        # Validamos los tipos de servicio
        if 'type' in df_clean.columns:
            valid_types = ['PS', 'PE', 'IN', 'NE', 'NW', 'TT']
            invalid_type = ~df_clean['type'].isin(valid_types)
            validation_issues['invalid_type'] = invalid_type.sum()
        
        # Validamos  outcomes
        if 'outcome' in df_clean.columns:
            valid_outcomes = ['AGENT', 'HANG', 'PHANTOM']
            invalid_outcome = ~df_clean['outcome'].isin(valid_outcomes)
            validation_issues['invalid_outcome'] = invalid_outcome.sum()
        
        # Validamos las fechas que estén en 1999
        if 'date' in df_clean.columns:
            invalid_year = df_clean['date'].dt.year != 1999
            validation_issues['invalid_year'] = invalid_year.sum()
            df_clean = df_clean[~invalid_year]
        
        self.cleaning_report['business_rule_violations'] = validation_issues
        
        return df_clean
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removemos outliers estadísticos."""
        df_clean = df.copy()
        outlier_info = {}
        
        # Removemos los outliers en tiempos usando IQR
        time_columns = ['vru_time', 'q_time', 'service_time']
        
        for col in time_columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR  
                upper_bound = Q3 + 3 * IQR
                
                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    outlier_info[col] = outlier_count
                    df_clean.loc[outliers, col] = pd.NA
        
        self.cleaning_report['outliers_removed'] = outlier_info
        
        return df_clean
    
    def _create_derived_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creamos las variables derivadas básicas."""
        df_clean = df.copy()
        
        # Extraemos la información de fecha
        if 'date' in df_clean.columns:
            df_clean['year'] = df_clean['date'].dt.year
            df_clean['month'] = df_clean['date'].dt.month
            df_clean['day'] = df_clean['date'].dt.day
            df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
            df_clean['week_of_year'] = df_clean['date'].dt.isocalendar().week
        
        # Variable indicadora de cliente identificado
        if 'customer_id' in df_clean.columns:
            df_clean['is_identified_customer'] = df_clean['customer_id'].notna()
        
        # Tiempo total de llamada
        time_cols = ['vru_time', 'q_time', 'service_time']
        if all(col in df_clean.columns for col in time_cols):
            df_clean['total_call_time'] = df_clean[time_cols].sum(axis=1, skipna=True)
        
        # Indicador de abandono
        if 'outcome' in df_clean.columns:
            df_clean['is_abandoned'] = df_clean['outcome'] == 'HANG'
            df_clean['is_served'] = df_clean['outcome'] == 'AGENT'
        
        return df_clean
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Obtenemos el reporte completo de limpieza."""
        return self.cleaning_report


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función wrapper para limpiar dataset del call center.
    
    Args:
        df (pd.DataFrame): Dataset crudo
        
    Returns:
        pd.DataFrame: Dataset limpio
    """
    cleaner = DataCleaner()
    return cleaner.clean_dataset(df)


def remove_outliers(df: pd.DataFrame, 
                   columns: List[str], 
                   method: str = 'iqr',
                   threshold: float = 3.0) -> pd.DataFrame:
    """
    Removemos los outliers de columnas específicas.
    
    Args:
        df (pd.DataFrame): Dataset
        columns (List[str]): Columnas a limpiar
        method (str): Método ('iqr' o 'zscore')
        threshold (float): Umbral para detección
        
    Returns:
        pd.DataFrame: Dataset sin outliers
    """
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            outliers = z_scores > threshold
            
        df_clean = df_clean[~outliers]
    
    return df_clean


if __name__ == "__main__":
    # Ejemplo de uso
    from data_ingestion import load_data
    
    try:
        # Cargamos los datos
        df = load_data()
        print(f"Datos originales: {df.shape}")
        
        # Limpiamos los datos
        df_clean = clean_dataset(df)
        print(f"Datos limpios: {df_clean.shape}")
        
        # Mostramos algunos datos estadísticos
        print("\nPrimeras 5 filas del dataset limpio:")
        print(df_clean.head())
        
    except Exception as e:
        print(f"Error: {e}")
