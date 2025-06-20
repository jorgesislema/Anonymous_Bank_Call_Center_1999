# 📖 Diccionario de Datos: call_center_clean.csv

| Columna        | Tipo         | Descripción                                                                                   | Ejemplo           |
|----------------|--------------|----------------------------------------------------------------------------------------------|-------------------|
| vru.line       | string       | Línea o canal de entrada VRU (IVR)                                                            | AA0101            |
| call_id        | int64        | Identificador único de la llamada                                                            | 33116             |
| customer_id    | float64      | Identificador del cliente (NaN o 0 para clientes anónimos)                                   | 9664491.0         |
| priority       | int8         | Prioridad de la llamada (0: baja, 1: media, 2: alta)                                         | 2                 |
| type           | string       | Tipo de servicio o transacción (PS, PE, IN, NE, NW, TT)                                      | PS                |
| date           | date         | Fecha de la llamada (YYYY-MM-DD)                                                             | 1999-01-01        |
| vru_entry      | string (hh:mm:ss) | Hora de entrada al sistema IVR (formato HH:MM:SS)                                         | 0:00:31           |
| vru_exit       | string (hh:mm:ss) | Hora de salida del IVR                                                                     | 0:00:36           |
| vru_time       | int64        | Tiempo en IVR en segundos                                                                    | 5                 |
| q_start        | string (hh:mm:ss) | Hora de inicio en cola                                                                     | 0:00:36           |
| q_exit         | string (hh:mm:ss) | Hora de salida de la cola                                                                  | 0:03:09           |
| q_time         | int64        | Tiempo en cola en segundos                                                                   | 153               |
| outcome        | string       | Resultado de la llamada (AGENT: atendida, HANG: abandono, PHANTOM: fantasma)                 | HANG              |
| ser_start      | string (hh:mm:ss) | Hora de inicio de servicio                                                                 | 0:00:00           |
| ser_exit       | string (hh:mm:ss) | Hora de fin de servicio                                                                   | 0:00:00           |
| ser_time       | int64        | Tiempo de servicio en segundos                                                               | 0                 |
| server         | string       | Nombre o identificador del agente/servidor (o NO_SERVER si no fue atendida)                  | NO_SERVER         |
| startdate      | int64        | Campo auxiliar (usualmente 0, puede indicar fecha/hora de inicio en otros formatos)           | 0                 |

**Notas:**
- Los campos de tiempo en formato string (hh:mm:ss) pueden ser "0:00:00" si no aplica.
- customer_id puede ser 0 o NaN para llamadas anónimas.
- outcome define el destino final de la llamada.
- Los tiempos (vru_time, q_time, ser_time) están en segundos.
