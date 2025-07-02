"""
Módulo de Feature Engineering
=============================

Este módulo contiene funciones para crear nuevas variables y características
que mejoren el análisis y modelado de los datos del call center.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, time
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Clase para crear features avanzadas para el análisis del call center."""
    
    def __init__(self):
        """Inicializar el feature engineer."""
        self.feature_info = {}
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear conjunto completo de features para análisis.
        
        Args:
            df (pd.DataFrame): Dataset limpio
            
        Returns:
            pd.DataFrame: Dataset con features adicionales
        """
        logger.info("Iniciando creación de features")
        
        df_featured = df.copy()
        
        # 1. Features temporales avanzadas
        df_featured = self._create_temporal_features(df_featured)
        
        # 2. Features de comportamiento del call center
        df_featured = self._create_call_center_features(df_featured)
        
        # 3. Features de eficiencia y performance
        df_featured = self._create_efficiency_features(df_featured)
        
        # 4. Features de segmentación de clientes
        df_featured = self._create_customer_features(df_featured)
        
        # 5. Features de análisis de carga de trabajo
        df_featured = self._create_workload_features(df_featured)
        
        # 6. Features estadísticas agregadas
        df_featured = self._create_aggregated_features(df_featured)
        
        # 7. Features categóricas codificadas
        df_featured = self._encode_categorical_features(df_featured)
        
        logger.info(f"Features creadas. Columnas: {df.shape[1]} -> {df_featured.shape[1]}")
        
        return df_featured
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear features temporales avanzadas."""
        df_featured = df.copy()
        
        if 'date' in df_featured.columns:
            # Features básicas de fecha
            df_featured['year'] = df_featured['date'].dt.year
            df_featured['month'] = df_featured['date'].dt.month
            df_featured['day'] = df_featured['date'].dt.day
            df_featured['day_of_week'] = df_featured['date'].dt.dayofweek
            df_featured['week_of_year'] = df_featured['date'].dt.isocalendar().week
            df_featured['quarter'] = df_featured['date'].dt.quarter
            
            # Features de calendario judío/israelí
            df_featured['is_weekend'] = df_featured['day_of_week'].isin([4, 5])  # Viernes y Sábado
            df_featured['is_sunday'] = df_featured['day_of_week'] == 6  # Domingo (primer día laboral)
            df_featured['is_thursday'] = df_featured['day_of_week'] == 3  # Último día laboral completo
            
            # Features estacionales
            df_featured['season'] = df_featured['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            })
            
            # Features de feriados (aproximados para Israel 1999)
            df_featured['is_holiday'] = self._identify_holidays(df_featured['date'])
            
        # Features de hora del día (parsing de time strings)
        time_columns = ['vru_entry', 'vru_exit', 'q_start', 'q_exit', 'service_start', 'service_exit']
        for col in time_columns:
            if col in df_featured.columns:
                hour_col = f'{col}_hour'
                df_featured[hour_col] = self._extract_hour_from_time(df_featured[col])
                
                # Categorizar en períodos del día
                if col == 'vru_entry':
                    df_featured['call_period'] = self._categorize_time_period(df_featured[hour_col])
        
        return df_featured
    
    def _identify_holidays(self, date_series: pd.Series) -> pd.Series:
        """Identificar feriados aproximados en Israel 1999."""
        holidays_1999 = [
            # Feriados fijos aproximados
            '1999-01-01',  # Año Nuevo
            '1999-04-30',  # Día de la Independencia
            '1999-05-10',  # Lag BaOmer
            '1999-09-11',  # Rosh Hashaná
            '1999-09-20',  # Yom Kippur
            '1999-12-25',  # Navidad (cristiana)
        ]
        
        holiday_dates = pd.to_datetime(holidays_1999)
        return date_series.isin(holiday_dates)
    
    def _extract_hour_from_time(self, time_series: pd.Series) -> pd.Series:
        """Extraer hora de strings de tiempo en formato HHMMSS."""
        def parse_hour(time_str):
            try:
                if pd.isna(time_str) or time_str == '0:00:00':
                    return np.nan
                
                time_str = str(time_str).strip()
                
                # Si es formato HHMMSS
                if len(time_str) == 6:
                    return int(time_str[:2])
                
                # Si ya está en formato HH:MM:SS
                if ':' in time_str:
                    return int(time_str.split(':')[0])
                
                return np.nan
                
            except:
                return np.nan
        
        return time_series.apply(parse_hour)
    
    def _categorize_time_period(self, hour_series: pd.Series) -> pd.Series:
        """Categorizar horas en períodos del día."""
        def get_period(hour):
            if pd.isna(hour):
                return 'Unknown'
            if 7 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            elif 21 <= hour <= 23 or 0 <= hour < 7:
                return 'Night'
            else:
                return 'Unknown'
        
        return hour_series.apply(get_period)
    
    def _create_call_center_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear features específicas del call center."""
        df_featured = df.copy()
        
        # Features de VRU
        if 'vru_line' in df_featured.columns:
            df_featured['vru_id'] = df_featured['vru_line'].str[:4] if df_featured['vru_line'].dtype == 'object' else None
            df_featured['line_number'] = df_featured['vru_line'].str[-2:] if df_featured['vru_line'].dtype == 'object' else None
        
        # Features de resultado de llamada
        if 'outcome' in df_featured.columns:
            df_featured['is_successful'] = df_featured['outcome'] == 'AGENT'
            df_featured['is_abandoned'] = df_featured['outcome'] == 'HANG'
            df_featured['is_phantom'] = df_featured['outcome'] == 'PHANTOM'
        
        # Features de prioridad del cliente
        if 'priority' in df_featured.columns:
            df_featured['is_high_priority'] = df_featured['priority'] == 2
            df_featured['is_regular_priority'] = df_featured['priority'].isin([0, 1])
        
        # Features de tipo de servicio
        if 'type' in df_featured.columns:
            df_featured['is_regular_service'] = df_featured['type'].isin(['PS', 'PE'])
            df_featured['is_internet_service'] = df_featured['type'] == 'IN'
            df_featured['is_prospect'] = df_featured['type'] == 'NW'
            df_featured['is_stock_service'] = df_featured['type'] == 'NE'
            df_featured['is_callback'] = df_featured['type'] == 'TT'
        
        # Features de agente
        if 'server' in df_featured.columns:
            df_featured['has_server'] = (df_featured['server'] != 'NO_SERVER') & df_featured['server'].notna()
            
        return df_featured
    
    def _create_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear features de eficiencia y performance."""
        df_featured = df.copy()
        
        # Ratios de tiempo
        time_cols = ['vru_time', 'q_time', 'service_time']
        if all(col in df_featured.columns for col in time_cols):
            total_time = df_featured[time_cols].sum(axis=1, skipna=True)
            df_featured['total_call_duration'] = total_time
            
            # Proporciones de tiempo
            df_featured['vru_time_ratio'] = df_featured['vru_time'] / (total_time + 1e-6)
            df_featured['queue_time_ratio'] = df_featured['q_time'] / (total_time + 1e-6)
            df_featured['service_time_ratio'] = df_featured['service_time'] / (total_time + 1e-6)
            
            # Eficiencia de procesamiento
            df_featured['processing_efficiency'] = df_featured['service_time'] / (df_featured['vru_time'] + df_featured['q_time'] + 1e-6)
        
        # Categorización de tiempos de espera
        if 'q_time' in df_featured.columns:
            df_featured['queue_wait_category'] = pd.cut(
                df_featured['q_time'], 
                bins=[0, 30, 120, 300, float('inf')],
                labels=['Short', 'Medium', 'Long', 'Very_Long'],
                include_lowest=True
            )
        
        # Categorización de tiempos de servicio
        if 'service_time' in df_featured.columns:
            df_featured['service_duration_category'] = pd.cut(
                df_featured['service_time'],
                bins=[0, 180, 600, 1200, float('inf')],
                labels=['Quick', 'Normal', 'Long', 'Extended'],
                include_lowest=True
            )
        
        return df_featured
    
    def _create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear features de segmentación de clientes."""
        df_featured = df.copy()
        
        if 'customer_id' in df_featured.columns:
            # Identificar clientes vs prospectos
            df_featured['is_identified_customer'] = df_featured['customer_id'].notna()
            df_featured['is_prospect'] = df_featured['customer_id'].isna()
            
            # Agregar features por cliente (se calculan por separado)
            customer_stats = self._calculate_customer_statistics(df_featured)
            df_featured = df_featured.merge(customer_stats, on='customer_id', how='left')
        
        return df_featured
    
    def _calculate_customer_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular estadísticas agregadas por cliente."""
        customer_stats = df.groupby('customer_id').agg({
            'call_id': 'count',  # Número de llamadas
            'service_time': ['mean', 'sum'],  # Tiempo promedio y total de servicio
            'q_time': 'mean',  # Tiempo promedio en cola
            'is_abandoned': 'mean',  # Tasa de abandono
            'priority': 'first'  # Prioridad del cliente
        }).round(2)
        
        # Aplanar nombres de columnas
        customer_stats.columns = [
            'customer_call_count', 'avg_service_time', 'total_service_time',
            'avg_queue_time', 'abandonment_rate', 'customer_priority'
        ]
        
        # Categorizar frecuencia de llamadas
        customer_stats['customer_frequency_category'] = pd.cut(
            customer_stats['customer_call_count'],
            bins=[0, 1, 3, 10, float('inf')],
            labels=['One_time', 'Occasional', 'Regular', 'Frequent']
        )
        
        return customer_stats.reset_index()
    
    def _create_workload_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear features de análisis de carga de trabajo."""
        df_featured = df.copy()
        
        # Features de carga por fecha y hora
        if 'date' in df_featured.columns and 'vru_entry_hour' in df_featured.columns:
            # Contar llamadas por día
            daily_calls = df_featured.groupby('date').size()
            df_featured['daily_call_volume'] = df_featured['date'].map(daily_calls)
            
            # Contar llamadas por hora del día
            hourly_calls = df_featured.groupby('vru_entry_hour').size()
            df_featured['hourly_call_volume'] = df_featured['vru_entry_hour'].map(hourly_calls)
            
            # Categorizar volumen de llamadas
            df_featured['volume_category'] = pd.cut(
                df_featured['daily_call_volume'],
                bins=[0, 50, 100, 200, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very_High']
            )
        
        return df_featured
    
    def _create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear features estadísticas agregadas."""
        df_featured = df.copy()
        
        # Features de agente (si hay información de servidor)
        if 'server' in df_featured.columns and df_featured['server'].notna().any():
            agent_stats = df_featured.groupby('server').agg({
                'service_time': ['mean', 'count'],
                'is_abandoned': 'mean'
            })
            
            agent_stats.columns = ['agent_avg_service_time', 'agent_call_count', 'agent_abandonment_rate']
            agent_stats = agent_stats.reset_index()
            
            df_featured = df_featured.merge(agent_stats, on='server', how='left')
        
        return df_featured
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Codificar variables categóricas."""
        df_featured = df.copy()
        
        # One-hot encoding para variables categóricas principales
        categorical_cols = ['type', 'outcome', 'season', 'call_period']
        
        for col in categorical_cols:
            if col in df_featured.columns:
                dummies = pd.get_dummies(df_featured[col], prefix=col, dummy_na=True)
                df_featured = pd.concat([df_featured, dummies], axis=1)
        
        return df_featured


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función wrapper para crear features.
    
    Args:
        df (pd.DataFrame): Dataset limpio
        
    Returns:
        pd.DataFrame: Dataset con features adicionales
    """
    engineer = FeatureEngineer()
    return engineer.create_features(df)


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crear solo features temporales.
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset con features temporales
    """
    engineer = FeatureEngineer()
    return engineer._create_temporal_features(df)


def create_customer_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crear segmentos de clientes basados en comportamiento.
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset con segmentos de cliente
    """
    df_segmented = df.copy()
    
    if 'customer_id' in df_segmented.columns:
        # Calcular métricas por cliente
        customer_metrics = df_segmented.groupby('customer_id').agg({
            'call_id': 'count',
            'service_time': 'mean',
            'q_time': 'mean',
            'is_abandoned': 'mean',
            'priority': 'first'
        }).round(2)
        
        customer_metrics.columns = ['call_frequency', 'avg_service_time', 'avg_queue_time', 'abandonment_rate', 'priority']
        
        # Crear segmentos simples
        def create_segment(row):
            if row['priority'] == 2:
                return 'VIP'
            elif row['call_frequency'] >= 5:
                return 'Frequent'
            elif row['abandonment_rate'] > 0.3:
                return 'Impatient'
            elif row['avg_service_time'] > 300:
                return 'Complex_Needs'
            else:
                return 'Regular'
        
        customer_metrics['customer_segment'] = customer_metrics.apply(create_segment, axis=1)
        
        # Mergear con dataset original
        df_segmented = df_segmented.merge(
            customer_metrics[['customer_segment']], 
            on='customer_id', 
            how='left'
        )
    
    return df_segmented


if __name__ == "__main__":
    # Ejemplo de uso
    from data_ingestion import load_data
    from data_cleaning import clean_dataset
    
    try:
        # Cargar y limpiar datos
        df = load_data()
        df_clean = clean_dataset(df)
        print(f"Datos limpios: {df_clean.shape}")
        
        # Crear features
        df_featured = create_features(df_clean)
        print(f"Datos con features: {df_featured.shape}")
        
        # Mostrar nuevas columnas
        new_columns = set(df_featured.columns) - set(df_clean.columns)
        print(f"\nNuevas features creadas ({len(new_columns)}):")
        for col in sorted(new_columns):
            print(f"  - {col}")
            
    except Exception as e:
        print(f"Error: {e}")
