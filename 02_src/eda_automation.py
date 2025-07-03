#!/usr/bin/env python3
"""
Automatización del Análisis Exploratorio de Datos del Call Center
================================================================

Este módulo automatiza la ejecución del análisis exploratorio para generar
insights clave sobre los datos del call center de forma programática.
"""
# Bibiotecas necesarias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import os
import json
from scipy import stats
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CallCenterEDA:
    """
    Clase para automatizar el análisis exploratorio de datos del call center
    """
    
    def __init__(self, data_path):
        """
        Inicializar la clase EDA
        
        Args:
            data_path (str): Ruta al archivo de datos limpios
        """
        self.data_path = data_path
        self.df = None
        self.insights = {}
        
        # Configuraciones
        warnings.filterwarnings('ignore')
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self):
        """Cargar datos limpios"""
        try:
            logger.info("Cargando datos limpios...")
            if self.data_path.endswith('.parquet'):
                self.df = pd.read_parquet(self.data_path)
            else:
                self.df = pd.read_csv(self.data_path)
            
            logger.info(f"Datos cargados exitosamente: {self.df.shape}")
            return True
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            return False
    
    def generate_summary_stats(self):
        """Generamos las estadísticas descriptivas avanzadas"""
        logger.info("Generando estadísticas descriptivas...")
        
        summary = {
            'shape': self.df.shape,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'numeric_summary': self.df.describe().to_dict()
        }
        
        self.insights['summary_stats'] = summary
        
        print("RESUMEN DE DATOS")
        print("=" * 50)
        print(f"Filas: {summary['shape'][0]:,}")
        print(f"Columnas: {summary['shape'][1]}")
        print(f"Memoria utilizada: {summary['memory_usage_mb']:.2f} MB")
        print(f"Valores faltantes: {sum(summary['missing_values'].values())}")
        
        return summary
    
    def analyze_temporal_patterns(self):
        """Analizamos los patrones temporales"""
        logger.info("Analizando patrones temporales...")
        
        # Convertimos date a datetime si no lo está
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            
            # Creamos las  características temporales
            self.df['day_of_week'] = self.df['date'].dt.day_name()
            self.df['month'] = self.df['date'].dt.month
            self.df['quarter'] = self.df['date'].dt.quarter
            
            # Análisis por día de la semana
            daily_stats = self.df.groupby('day_of_week').agg({
                'outcome': 'count',
                'ser_time': 'mean',
                'vru_time': 'mean'
            }).round(2)
            
            print("\nANÁLISIS TEMPORAL")
            print("=" * 50)
            print("Llamadas por día de la semana:")
            print(daily_stats['outcome'].sort_values(ascending=False))
            
            self.insights['temporal'] = {
                'daily_stats': daily_stats.to_dict(),
                'total_days': self.df['date'].nunique(),
                'date_range': {
                    'start': str(self.df['date'].min()),
                    'end': str(self.df['date'].max())
                }
            }
    
    def analyze_call_center_behavior(self):
        """Analizar comportamiento del call center"""
        logger.info("Analizando comportamiento del call center...")
        
        # Análisis de outcomes
        outcome_stats = self.df['outcome'].value_counts()
        outcome_pct = self.df['outcome'].value_counts(normalize=True) * 100
        
        # Análisis de tipos de servicio
        if 'ser_start' in self.df.columns:
            service_stats = self.df['ser_start'].value_counts()
        
        # Análisis de tiempos
        time_stats = {
            'ser_time': {
                'mean': self.df['ser_time'].mean(),
                'median': self.df['ser_time'].median(),
                'std': self.df['ser_time'].std()
            },
            'vru_time': {
                'mean': self.df['vru_time'].mean(),
                'median': self.df['vru_time'].median(),
                'std': self.df['vru_time'].std()
            }
        }
        
        print("\n ANÁLISIS DE CALL CENTER")
        print("=" * 50)
        print("Distribución de resultados:")
        for outcome, count in outcome_stats.items():
            pct = outcome_pct[outcome]
            print(f"  {outcome}: {count:,} ({pct:.1f}%)")
        
        print(f"\nTiempo promedio de servicio: {time_stats['ser_time']['mean']:.2f}s")
        print(f"Tiempo promedio en VRU: {time_stats['vru_time']['mean']:.2f}s")
        
        self.insights['call_center'] = {
            'outcome_distribution': outcome_stats.to_dict(),
            'outcome_percentages': outcome_pct.to_dict(),
            'time_statistics': time_stats
        }
    
    def create_visualizations(self, save_dir=None):
        """Crear visualizaciones clave"""
        logger.info("Creando visualizaciones...")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 1. Distribución de outcomes
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        outcome_counts = self.df['outcome'].value_counts()
        plt.pie(outcome_counts.values, labels=outcome_counts.index, autopct='%1.1f%%')
        plt.title('Distribución de Resultados de Llamadas')
        
        # 2. Histograma de tiempos de servicio
        plt.subplot(2, 2, 2)
        plt.hist(self.df['ser_time'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Tiempo de Servicio (segundos)')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Tiempos de Servicio')
        
        # 3. Análisis temporal
        if 'day_of_week' in self.df.columns:
            plt.subplot(2, 2, 3)
            daily_calls = self.df['day_of_week'].value_counts()
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_calls = daily_calls.reindex([day for day in days_order if day in daily_calls.index])
            
            plt.bar(range(len(daily_calls)), daily_calls.values)
            plt.xticks(range(len(daily_calls)), [day[:3] for day in daily_calls.index], rotation=45)
            plt.xlabel('Día de la Semana')
            plt.ylabel('Número de Llamadas')
            plt.title('Llamadas por Día de la Semana')
        
        # 4. Box plot de tiempos por outcome
        plt.subplot(2, 2, 4)
        self.df.boxplot(column='ser_time', by='outcome', ax=plt.gca())
        plt.xticks(rotation=45)
        plt.title('Tiempo de Servicio por Resultado')
        plt.suptitle('')  # Remover el título automático
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'eda_overview.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Visualización guardada en {save_dir}")
        
        plt.show()
    
    def generate_correlation_analysis(self):
        """Analizar correlaciones entre variables numéricas"""
        logger.info("Analizando correlaciones...")
        
        # Seleccionar columnas numéricas
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            correlation_matrix = self.df[numeric_cols].corr()
            
            # Creamos heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Matriz de Correlación - Variables Numéricas')
            plt.tight_layout()
            plt.show()
            
            self.insights['correlations'] = correlation_matrix.to_dict()
    
    def detect_outliers(self):
        """Detectamos los valores atípicos en variables clave"""
        logger.info("Detectando valores atípicos...")
        
        outliers_summary = {}
        
        # Columnas numéricas para análisis de outliers
        key_columns = ['ser_time', 'vru_time']
        
        for col in key_columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(self.df)) * 100
                
                outliers_summary[col] = {
                    'count': outlier_count,
                    'percentage': outlier_percentage,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
                
                print(f"\nOUTLIERS en {col}:")
                print(f"  Cantidad: {outlier_count:,} ({outlier_percentage:.2f}%)")
                print(f"  Límites: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        self.insights['outliers'] = outliers_summary
    
    def run_complete_analysis(self, save_dir=None):
        """Ejecutar análisis completo"""
        logger.info("Iniciando análisis exploratorio completo...")
        
        # Cargamos los datos
        if not self.load_data():
            return False
        
        # Ejecutamos todos los análisis
        self.generate_summary_stats()
        self.analyze_temporal_patterns()
        self.analyze_call_center_behavior()
        self.generate_correlation_analysis()
        self.detect_outliers()
        self.create_visualizations(save_dir)
        
        # Guardamos los insights
        if save_dir:
            insights_file = os.path.join(save_dir, 'eda_insights.json')
            with open(insights_file, 'w', encoding='utf-8') as f:
                json.dump(self.insights, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Insights guardados en {insights_file}")
        
        logger.info("Análisis exploratorio completado exitosamente")
        return True
    
    def print_summary_report(self):
        """Imprimir reporte resumen del análisis"""
        print("\n" + "="*60)
        print(" REPORTE RESUMEN - ANÁLISIS EXPLORATORIO")
        print("="*60)
        
        if 'summary_stats' in self.insights:
            stats = self.insights['summary_stats']
            print(f"Dataset: {stats['shape'][0]:,} filas x {stats['shape'][1]} columnas")
            print(f"Memoria: {stats['memory_usage_mb']:.2f} MB")
        
        if 'temporal' in self.insights:
            temporal = self.insights['temporal']
            print(f"Período analizado: {temporal['date_range']['start']} - {temporal['date_range']['end']}")
            print(f"Total de días: {temporal['total_days']}")
        
        if 'call_center' in self.insights:
            cc = self.insights['call_center']
            print(f"Tiempo promedio de servicio: {cc['time_statistics']['ser_time']['mean']:.2f}s")
            print(f"Tiempo promedio en VRU: {cc['time_statistics']['vru_time']['mean']:.2f}s")
        
        print("="*60)


def main():
    """Función principal para ejecutar el análisis"""
    
    # Rutas
    data_path = "../00_data/processed/call_center_clean.parquet"
    output_dir = "../03_outputs/eda_results"
    
    # Creamos instancia de EDA
    eda = CallCenterEDA(data_path)
    
    # Ejecutamos el análisis completo
    success = eda.run_complete_analysis(save_dir=output_dir)
    
    if success:
        eda.print_summary_report()
    else:
        print("❌ Error en el análisis exploratorio")


if __name__ == "__main__":
    main()
