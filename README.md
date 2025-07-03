# Análisis Integral del Call Center - Anonymous Bank

## Contexto Histórico

El conjunto de datos fue meticulosamente recopilado en el call center de un banco israelí, denominado "Anonymous Bank" para proteger su identidad, y cubre todas las llamadas registradas durante el año completo de 1999, desde el 1 de enero hasta el 31 de diciembre. Fue puesto a disposición del público para fines de investigación por Ilan Guedj y Avi Mandelbaum, de la Facultad de Ingeniería Industrial y Gestión de la Universidad Technion en Haifa, Israel. Este origen académico es fundamental para entender su amplio uso y el rigor de los análisis que ha inspirado.

La escala del conjunto de datos es notable. Contiene registros de aproximadamente 450,000 llamadas, que se desglosan más específicamente en 444,431 eventos distribuidos en 54,471 casos de clientes. Este volumen proporciona una base estadísticamente robusta para realizar análisis detallados y extraer conclusiones fiables sobre el comportamiento del sistema.

### El Flujo Documentado del Cliente

La documentación y los estudios posteriores permiten reconstruir con precisión el viaje de un cliente a través del sistema del "Anonymous Bank":

1. **Entrada al Sistema**: Un cliente llama a uno de los varios números de teléfono del banco, lo que lo conecta inmediatamente con una Unidad de Respuesta de Voz (VRU), la puerta de entrada automatizada del sistema.
2. **Interacción con la VRU**: Dentro de la VRU, el cliente tiene dos opciones principales: completar su transacción de forma totalmente automatizada o indicar la necesidad de hablar con un agente humano.
3. **Transferencia Directa**: Si un agente está disponible en el momento en que el cliente solicita asistencia humana, la llamada se transfiere directamente. En este escenario ideal, el tiempo de espera en cola es cero.
4. **Entrada a la Cola**: Si todos los agentes están ocupados, la llamada se coloca en una tele-cola. Esta cola opera bajo un principio estricto de Primero en Entrar, Primero en Salir (FCFS). El registro de tiempo `q_start` marca el momento exacto en que la llamada entra en esta fase de espera.
5. **Abandono Potencial**: Mientras esperan en la cola, los clientes pueden optar por colgar antes de ser atendidos por un agente. Este evento se conoce como abandono y es una métrica crítica de la insatisfacción del cliente.
6. **Llegada al Servicio Humano**: El registro de tiempo `vru_exit` es un punto de datos crucial. Marca el momento en que el cliente sale del sistema automatizado, ya sea para hablar inmediatamente con un agente o para unirse a la cola. Desde la perspectiva del análisis de colas, este es el momento de "llegada" al sistema de servicio humano.

### Estructura de Datos y Variables Clave

Los datos están organizados en archivos de texto mensuales, lo que facilita su manejo. Un detalle operativo crucial es el horario del call center: de domingo a jueves, opera de 7:00 a 24:00. El horario se reduce los viernes, cerrando a las 14:00, y se reanuda el sábado por la noche, alrededor de las 20:00, en observancia del Sabbat. Esta variabilidad es un factor esencial para cualquier análisis de series temporales.

La siguiente tabla consolida las variables más importantes del conjunto de datos, proporcionando un diccionario unificado para comprender los análisis que se presentan a continuación.

| Nombre de la Variable | Descripción | Tipo | Valores Posibles |
|-----------------------|-------------|------|------------------|
| `vru.line`           | Identificador de 6 dígitos para la VRU y la línea específicas por las que entró la llamada. | String | AA01-1 a AA06-16 |
| `call.id`            | Un identificador único asignado a cada llamada entrante. | Integer | 5 dígitos |
| `customer.id`        | Un identificador único para un cliente existente. Un valor de cero indica un no-cliente o un llamante no identificado. | Integer | 0-12 dígitos |
| `priority`           | Nivel de prioridad del cliente: 0 o 1 para clientes no identificados/regulares, 2 para clientes prioritarios que obtienen una posición avanzada en la cola. | Integer | 0, 1, 2 |
| `type`               | El tipo de servicio solicitado, p. ej., 'PS' para 'Peilut Shotefet' (actividad regular). | String | PS, PE, IN, NE, NW, TT |
| `date`               | La fecha de la llamada en formato 99MMDD. | Date | YYMMDD |
| `vru_entry`          | La hora en que la llamada entró con éxito en la VRU (hh:mm:ss). | Time | HHMMSS |
| `vru_exit`           | La hora en que la llamada salió de la VRU para dirigirse a un agente o a una cola. | Time | HHMMSS |
| `vru_time`           | El tiempo calculado en segundos que se pasó en la VRU. | Integer | 1-999 |
| `q_start`            | La hora en que la llamada se unió a la cola (00:00:00 si no hubo espera en cola). | Time | HHMMSS |
| `q_exit`             | La hora en que la llamada salió de la cola (ya sea atendida o abandonada). | Time | HHMMSS |
| `q_time`             | El tiempo calculado en segundos que se pasó en la cola. | Integer | 1-999 |
| `outcome`            | El resultado de la llamada (p. ej., Atendida, Abandonada). | String | AGENT, HANG, PHANTOM |
| `ser_start`          | La hora en que el agente comenzó el servicio. | Time | HHMMSS |
| `ser_exit`           | La hora en que el agente finalizó el servicio. | Time | HHMMSS |
| `ser_time`           | El tiempo calculado en segundos que el agente dedicó al servicio. | Integer | 1-999 |
| `day.of.week`        | El día de la semana, añadido por conveniencia en algunos análisis. | Integer | 0-6 |

---

## Descripción del Proyecto

Este proyecto realiza un análisis exhaustivo de los datos del call center del "Anonymous Bank" correspondientes al año 1999. Implementamos un pipeline completo de ETL (Extract, Transform, Load), EDA (Exploratory Data Analysis) y modelado predictivo para optimizar las operaciones del call center.

## Objetivos del Proyecto

### Objetivos Principales
- **Análisis Operacional**: Identificar patrones en volumen de llamadas, tiempos de espera y eficiencia del servicio
- **Optimización de Recursos**: Determinar la distribución óptima de agentes por horarios y días
- **Predicción de Demanda**: Desarrollar modelos para predecir picos de llamadas
- **Análisis de Satisfacción**: Evaluar tasas de abandono y factores que las influencian
- **Segmentación de Clientes**: Clasificar clientes según patrones de uso y prioridad

### Objetivos Secundarios
- **Dashboard Interactivo**: Crear visualizaciones en Power BI para monitoreo en tiempo real
- **Automatización ETL**: Desarrollar pipeline automatizado para procesamiento de datos
- **Modelos Predictivos**: Implementar ML para forecasting y clasificación
- **Recomendaciones Estratégicas**: Proponer mejoras operacionales basadas en datos

## Arquitectura del Proyecto

```
call_center_analysis/
├── 00_data/                    # Datos del proyecto
│   ├── raw/                    # Datos originales sin procesar
│   │   └── Call_Center_1999_DataSet.csv
│   ├── processed/              # Datos limpios y transformados
│   │   ├── clean_call_data.csv
│   │   ├── aggregated_metrics.csv
│   │   └── featured_dataset.csv
│   └── external/               # Datos externos y maestros
│       ├── holidays_israel_1999.csv
│       ├── agent_shifts.csv
│       └── customer_segments.csv
│
├── 01_notebooks/               # Jupyter Notebooks para análisis
│   ├── sandbox/                # Experimentos y pruebas
│   ├── 01_initial_data_exploration.ipynb
│   ├── 02_data_cleaning_transformation.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_statistical_analysis.ipynb
│   ├── 05_machine_learning_models.ipynb
│   └── 06_final_insights_recommendations.ipynb
│
├── 02_src/                     # Código fuente modularizado
│   ├── __init__.py
│   ├── data_ingestion.py       # Funciones para cargar datos
│   ├── data_cleaning.py        # Limpieza y validación
│   ├── feature_engineering.py  # Creación de variables
│   ├── analysis.py             # Funciones de análisis
│   ├── modeling.py             # Modelos de ML
│   ├── visualization.py        # Gráficos y visualizaciones
│   └── utils.py                # Utilidades generales
│
├── 03_dashboards/              # Dashboards y aplicaciones
│   ├── power_bi/
│   │   ├── call_center_dashboard.pbix
│   │   └── data_model.json
│   ├── streamlit/
│   │   ├── app.py
│   │   └── components/
│   └── dash/
│       └── interactive_dashboard.py
│
│
├── tests/                      # Tests unitarios
│   ├── test_data_cleaning.py
│   ├── test_feature_engineering.py
│   └── test_models.py
│
├── config/                     # Configuraciones
│   ├── config.yaml
│   └── logging_config.py
│
├── requirements.txt            # Dependencias de Python
├── setup.py                   # Configuración del paquete
├── .env.template              # Variables de entorno
├── .gitignore                 # Archivos a ignorar en git
└── README.md                  # Este archivo
```

## Descripción del Dataset

### Información General
- **Período**: 1 de enero de 1999 - 31 de diciembre de 1999
- **Volumen**: ~20,000-30,000 llamadas por mes
- **Registros**: Cada fila representa una llamada individual
- **Campos**: 17 columnas con información detallada de cada llamada

### Estructura de Datos

| Campo | Descripción | Tipo | Valores Posibles |
|-------|-------------|------|------------------|
| `vru+line` | ID del VRU y línea (AA01-AA06, líneas 1-16) | String | AA01-1 a AA06-16 |
| `call_id` | Identificador único de llamada | Integer | 5 dígitos |
| `customer_id` | ID del cliente (0 si no identificado) | Integer | 0-12 dígitos |
| `priority` | Prioridad del cliente | Integer | 0=Regular, 1=Regular, 2=Alta |
| `type` | Tipo de servicio | String | PS, PE, IN, NE, NW, TT |
| `date` | Fecha de la llamada | Date | YYMMDD |
| `vru_entry` | Hora de entrada al VRU | Time | HHMMSS |
| `vru_exit` | Hora de salida del VRU | Time | HHMMSS |
| `vru_time` | Tiempo en VRU (segundos) | Integer | 1-999 |
| `q_start` | Hora de entrada a cola | Time | HHMMSS |
| `q_exit` | Hora de salida de cola | Time | HHMMSS |
| `q_time` | Tiempo en cola (segundos) | Integer | 1-999 |
| `outcome` | Resultado de la llamada | String | AGENT, HANG, PHANTOM |
| `ser_start` | Inicio del servicio | Time | HHMMSS |
| `ser_exit` | Fin del servicio | Time | HHMMSS |
| `ser_time` | Duración del servicio (segundos) | Integer | 1-999 |
| `server` | Agente que atendió | String | Nombre o NO_SERVER |

### Tipos de Servicio
- **PS**: Actividad Regular
- **PE**: Actividad Regular en Inglés
- **IN**: Consulta por Internet/Home Banking
- **NE**: Actividad por Acciones (Bolsa)
- **NW**: Cliente Prospecto
- **TT**: Callback solicitado por cliente

## Instalación y Configuración

### Prerrequisitos
- Python 3.8+
- Git
- Power BI Desktop (para dashboards)
- Jupyter Notebook/Lab

### Instalación

1. **Clonar el repositorio**
```bash
git clone <repository-url>
cd call_center_analysis
```

2. **Crear entorno virtual**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**
```bash
cp .env.template .env
# Editar .env con tus configuraciones
```

5. **Ejecutar tests**
```bash
pytest tests/
```

## Metodología de Análisis

### Fase 1: ETL (Extract, Transform, Load)
1. **Extracción**: Carga de datos desde CSV
2. **Transformación**: 
   - Limpieza de datos faltantes y outliers
   - Conversión de tipos de datos
   - Creación de variables derivadas
3. **Carga**: Almacenamiento en formatos optimizados

### Fase 2: EDA (Exploratory Data Analysis)
1. **Análisis Descriptivo**: Estadísticas básicas y distribuciones
2. **Análisis Temporal**: Patrones por hora, día, mes
3. **Análisis de Servicios**: Performance por tipo de servicio
4. **Análisis de Agentes**: Eficiencia y carga de trabajo

### Fase 3: Análisis Avanzado
1. **Análisis Estadístico**: Tests de hipótesis y correlaciones
2. **Segmentación**: Clustering de clientes y llamadas
3. **Modelado Predictivo**: Forecasting y clasificación
4. **Optimización**: Simulación de escenarios

## Herramientas y Tecnologías

### Análisis de Datos
- **Python**: Lenguaje principal
- **Pandas**: Manipulación de datos
- **NumPy**: Cálculos numéricos
- **SciPy**: Análisis estadístico

### Visualización
- **Matplotlib/Seaborn**: Gráficos estáticos
- **Plotly**: Visualizaciones interactivas
- **Power BI**: Dashboard ejecutivo
- **Streamlit/Dash**: Aplicaciones web

### Machine Learning
- **Scikit-learn**: Modelos tradicionales
- **XGBoost/LightGBM**: Boosting algorithms
- **Statsmodels**: Análisis estadístico avanzado

## Análisis Propuestos

### 1. Análisis Operacional
- Distribución de llamadas por hora/día/mes
- Tiempos promedio de espera y servicio
- Tasas de abandono por período
- Eficiencia por agente y VRU

### 2. Análisis de Clientes
- Segmentación por comportamiento
- Análisis de clientes de alta prioridad
- Patrones de uso por tipo de servicio
- Customer Journey mapping

### 3. Análisis Predictivo
- Forecasting de volumen de llamadas
- Predicción de tasas de abandono
- Optimización de staffing
- Detección de anomalías

### 4. Análisis de Calidad
- Service Level Agreements (SLA)
- Factores que afectan satisfacción
- Comparación de performance entre agentes
- Identificación de mejores prácticas

## KPIs y Métricas Clave

### Métricas Operacionales
- **Average Handle Time (AHT)**: Tiempo promedio de manejo
- **First Call Resolution (FCR)**: Resolución en primera llamada
- **Service Level**: % llamadas atendidas en tiempo objetivo
- **Abandonment Rate**: Tasa de abandono

### Métricas de Eficiencia
- **Agent Utilization**: Utilización de agentes
- **Queue Time**: Tiempo promedio en cola
- **VRU Efficiency**: Eficiencia del sistema VRU
- **Peak Hour Analysis**: Análisis de horas pico

### Métricas de Calidad
- **Customer Satisfaction**: Satisfacción del cliente
- **Call Quality Score**: Puntuación de calidad
- **Repeat Call Rate**: Tasa de llamadas repetidas
- **Agent Performance**: Performance por agente

## Fuentes de Datos Adicionales Requeridas

### Datos Internos Necesarios
1. **Maestro de Agentes**
   - ID, nombre, turno, habilidades
   - Fecha de ingreso, capacitación
   - Performance histórico

2. **Calendario de Feriados**
   - Feriados nacionales en Israel 1999
   - Días especiales bancarios
   - Horarios especiales

3. **Información de Clientes**
   - Demografía, segmento, antigüedad
   - Productos contratados
   - Historial de transacciones

### Datos Externos Sugeridos
1. **Datos Económicos**
   - Indicadores económicos de Israel 1999
   - Eventos del mercado financiero
   - Noticias relevantes del sector bancario

2. **Datos de Competencia**
   - Benchmarks de la industria
   - Estándares de servicio
   - Mejores prácticas del sector

## Pasos para Ejecutar el Análisis

### 1. Preparación Inicial
```bash
# Activar entorno y instalar dependencias
pip install -r requirements.txt

# Ejecutar notebook inicial
jupyter notebook 01_notebooks/01_initial_data_exploration.ipynb
```

### 2. Pipeline ETL
```bash
# Ejecutar limpieza de datos
python 02_src/data_cleaning.py

# Ejecutar feature engineering
python 02_src/feature_engineering.py
```

### 3. Análisis Exploratorio
```bash
# Ejecutar análisis completo
jupyter notebook 01_notebooks/03_exploratory_data_analysis.ipynb
```

### 4. Dashboard
```bash
# Ejecutar dashboard Streamlit
streamlit run 03_dashboards/streamlit/app.py
```

## Equipo y Contribuidores

### Roles Sugeridos
- **Data Scientist**: Análisis y modelado
- **Data Engineer**: Pipeline ETL
- **Business Analyst**: Interpretación de resultados
- **Visualization Specialist**: Dashboards y reportes

##  Contacto y Soporte

Para preguntas o sugerencias sobre este proyecto:
- 📧 Email: [tu-email@empresa.com]
- 💬 Slack: #call-center-analysis
- Wiki: [Link a documentación interna]

## Licencia

Este proyecto es propiedad de [Tu Empresa] y está sujeto a las políticas internas de uso de datos.

---

**Última actualización**: Mayo 2025
**Versión**: 1.0.0
**Estado**: En desarrollo activoCall_Center
Caso de negocio basado en un Call Center de un Banco: “Anonymous Bank” en Israel. El dataset contiene las llamadas registradas durante 12 meses (desde el 01/01/99 hasta el 31/12/99)
