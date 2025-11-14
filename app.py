# --------------------------------------------------------------------------
# app.py
#
# Aplicación Principal del Dashboard "Reefer-Tech" con Streamlit.
#
# v39 (Solución Profesional de Webhooks y KPIs)
# - CORREGIDO (CRÍTICO - WEBHOOKS): Eliminada la dependencia de Flask/Threading.
#   Se elimina la llamada a `start_webhook_thread`. Se añade
#   `st.session_state.previous_live_stats` y una nueva función
#   `check_and_log_alerts` que se ejecuta en cada ciclo y compara
#   el estado de `checkEngineLightIsOn` para registrar alertas
#   internamente. Esto soluciona los errores 303 y "Address in use".
# - CORREGIDO (CRÍTICO - KPIs ATORADOS): Los KPIs de Temperatura, Humedad
#   y Batería ahora toman su valor principal y timestamp desde el
#   último punto (`.iloc[-1]`) de su respectivo dataframe de
#   historial (`df_history_1h` o `df_battery_history_24h`),
#   asegurando que siempre estén tan actualizados como sus gráficos.
# - MEJORADO (UI PUERTA): `render_door_status` ahora busca el
#   último evento *diferente* en el historial de 1h y lo muestra
#   debajo del ping (ej. "Previo: Abierta a las 07:35:12").
# --------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium.features import DivIcon # <-- Ícono de pulso
from streamlit_folium import st_folium
import pytz
from datetime import datetime, timedelta
import json
import os

# Importar nuestras utilidades de backend (API, IA, Webhook)
import utils

# Importar el componente de auto-refresco
from streamlit_autorefresh import st_autorefresh

# --- 1. CONFIGURACIÓN INICIAL DE LA PÁGINA Y DEL ESTADO ---

st.set_page_config(
    page_title="Samsara Reefer Dashboard",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MEXICO_TZ = pytz.timezone("America/Mexico_City") 

# Intervalo de refresco
REFRESH_INTERVAL_SEC = 30

# --- INICIALIZACIÓN DE SESSION_STATE ---
try:
    if 'api_client' not in st.session_state:
        st.session_state.api_client = utils.SamsaraAPIClient(api_key=utils.SAMSARA_API_KEY)
    if 'ai_model' not in st.session_state:
        st.session_state.ai_model = utils.AIModels()
    
    # (v39) Hilo de Webhook ELIMINADO
    # if 'webhook_thread_started' not in st.session_state:
    #     utils.start_webhook_thread()
    #     st.session_state.webhook_thread_started = True

except ValueError as e:
    st.error(f"Error de Configuración: {e}. Revisa tu archivo .env.")
    st.stop()
except Exception as e:
    st.error(f"Error fatal al inicializar: {e}")
    st.stop

# Inicializar estado de la UI
if 'selected_vehicle_name' not in st.session_state:
    st.session_state.selected_vehicle_name = None
if 'selected_vehicle_obj' not in st.session_state:
    st.session_state.selected_vehicle_obj = None
if 'sensor_config' not in st.session_state:
    st.session_state.sensor_config = None
if 'last_webhook_timestamp' not in st.session_state:
    st.session_state.last_webhook_timestamp = None
    
# (v39) Estado para el nuevo motor de alertas
if 'previous_live_stats' not in st.session_state:
    st.session_state.previous_live_stats = None


# --- CSS v39 (Añadido .door-prev-event) ---
st.markdown("""
<style>
    .block-container { 
        padding-top: 2rem; padding-bottom: 2rem; 
        padding-left: 3rem; padding-right: 3rem;
    }
    .vehicle-info-header { 
        font-size: 0.9rem; color: #B0B0B0; 
        margin-bottom: -5px;
    }
    .vehicle-info-value {
        font-size: 1.1rem; color: #FAFAFA; font-weight: 600;
        font-family: 'Consolas', 'Monaco', monospace;
    }
    .sensor-list-header {
        font-size: 1.0rem; font-weight: 600; color: #FAFAFA;
        margin-bottom: 5px;
    }
    .sensor-list-item {
        font-size: 0.85rem; color: #B0B0B0; margin-left: 10px;
        font-family: 'Consolas', 'Monaco', monospace;
    }
    .sensor-separator {
        margin-top: 10px; margin-bottom: 10px;
        border-top: 1px solid #262730;
    }
    
    .kpi-box {
        background-color: #0E1117; 
        border: 1px solid #262730; 
        border-radius: 8px;
        padding: 1rem;
        height: 100%;
    }
    .kpi-box .stMetric {
        border-bottom: 1px solid #262730; /* Borde inferior para KPI */
        padding-bottom: 0.5rem;
    }
    .kpi-box .stMetric label[data-testid="stMetricLabel"] { 
        font-size: 0.9rem; 
        color: #FAFAFA;
        font-weight: 600;
    }
    .kpi-box .stMetric p[data-testid="stMetricValue"] { font-size: 1.75rem; }

    .door-status-box {
        padding: 1rem; border-radius: 8px; text-align: center;
        border: 1px solid;
        height: 100%; 
        display: flex;
        flex-direction: column;
        justify-content: center; /* (v35) Centrar verticalmente */
    }
    .door-status-closed {
        background-color: rgba(46, 204, 113, 0.1);
        border-color: #2ECC71;
    }
    .door-status-open {
        background-color: rgba(231, 76, 60, 0.1);
        border-color: #E74C3C;
    }
    .door-status-text {
        font-size: 2.25rem; font-weight: 700;
        letter-spacing: 0.5px;
        padding-bottom: 0.5rem; 
    }
    .door-ping-time {
        font-size: 0.9rem; 
        color: #B0B0B0;
    }
    /* (v39) Estilo para el evento previo de puerta */
    .door-prev-event {
        font-size: 0.8rem;
        color: #808080;
        margin-top: 8px;
        font-style: italic;
    }
    
    .webhook-alert {
        border: 1px solid #262730; border-radius: 8px;
        padding: 0.75rem 1rem; margin-bottom: 0.5rem;
        background-color: #0E1117;
    }
    .webhook-alert-header { font-size: 0.9rem; color: #B0B0B0; }
    .webhook-alert-header strong { color: #FAFAFA; }
    .webhook-alert-message { font-size: 1rem; color: #FAFAFA; }
</style>
""", unsafe_allow_html=True)


# --- 2. TÍTULO Y AUTO-REFRESCO ---

st.title("❄️ Samsara Reefer-Tech")
st.caption(f"Monitoreo en tiempo real de temperatura, puertas y GPS. (Refresca cada {REFRESH_INTERVAL_SEC}s)")

st_autorefresh(interval=REFRESH_INTERVAL_SEC * 1000, limit=None, key="data_refresher")


# --- 3. FUNCIONES DE CARGA DE DATOS (CACHEADAS) ---

@st.cache_data(ttl=600)
def load_vehicle_list(_api_client):
    if not _api_client: return [], {}
    vehicles = _api_client.get_vehicles()
    vehicle_map_obj = {v['name']: v for v in vehicles}
    vehicle_names = [v['name'] for v in vehicles]
    return vehicle_names, vehicle_map_obj

@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def fetch_live_kpis(_api_client, sensor_config, vehicle_id):
    """(v39) Obtiene Snapshots (KPIs en vivo y estado de fallas)."""
    if not _api_client or not vehicle_id: return None, [], []
    stats_data = _api_client.get_live_stats(vehicle_id)
    temp_data, hum_data = _api_client.get_live_sensor_kpis(sensor_config, vehicle_id)
    return stats_data, temp_data, hum_data

@st.cache_data(ttl=REFRESH_INTERVAL_SEC) 
def fetch_live_sensor_history(_api_client, sensor_config, window_minutes, step_seconds):
    """(v37) Esta función es solo para SENSORES (Temp, Hum, Puerta)."""
    if not _api_client or not sensor_config:
        return [], {}
    return _api_client.get_live_sensor_history(
        sensor_config, 
        window_minutes, 
        step_seconds
    )

@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def fetch_vehicle_stats_history(_api_client, vehicle_id, window_minutes):
    """(v37) NUEVA función para obtener historial de Batería, GPS y Fallas."""
    if not _api_client or not vehicle_id:
        return None
    return _api_client.get_vehicle_stats_history(
        vehicle_id,
        window_minutes
    )

@st.cache_data
def process_sensor_history_data(results, column_map, step_seconds=30):
    """(v37) Renombrada para claridad. Procesa /sensors/history."""
    if not results or not column_map:
        return pd.DataFrame() 

    data_rows = []
    has_temp = "temperature" in column_map.values()
    has_hum = "humidity" in column_map.values()
    has_door = "doorClosed" in column_map.values()

    for point in results:
        # (v38) timeMs es naive, así que tz_localize es CORRECTO aquí
        row = {'timestamp': pd.to_datetime(point['timeMs'], unit='ms')}
        temp_val = None
        for i, value in enumerate(point['series']):
            col_name = column_map.get(i)
            if col_name and value is not None:
                if col_name == 'temperature':
                    temp_val = float(value) / 1000.0
                elif col_name == 'humidity':
                    row['humidity'] = float(value)
                elif col_name == 'doorClosed':
                    row['doorClosed'] = 1 if value else 0
        if temp_val is not None:
            row['temperature'] = temp_val
        data_rows.append(row)
        
    if not data_rows: 
        return pd.DataFrame()
    
    df = pd.DataFrame(data_rows).set_index('timestamp').sort_index()
    if has_temp and 'temperature' not in df.columns: df['temperature'] = np.nan
    if has_hum and 'humidity' not in df.columns: df['humidity'] = np.nan
    if has_door and 'doorClosed' not in df.columns: df['doorClosed'] = np.nan
    if df.empty:
        return pd.DataFrame()

    # (v38) tz_localize es CORRECTO aquí porque timeMs es naive
    df = df.tz_localize(pytz.utc).tz_convert(MEXICO_TZ)
    
    continuous_cols = [col for col in ['temperature', 'humidity'] if col in df.columns]
    df_continuous = df[continuous_cols].interpolate(method='time')
    discrete_cols = [col for col in ['doorClosed'] if col in df.columns]
    df_discrete = df[discrete_cols].ffill().bfill() 
    df = pd.concat([df_continuous, df_discrete], axis=1)
    
    return df.ffill().bfill()

@st.cache_data
def process_vehicle_stats_history(stats_history_data):
    """(v37) NUEVA función para procesar la respuesta de /stats/history."""
    df_battery = pd.DataFrame()
    all_faults_list = []

    if not stats_history_data or 'data' not in stats_history_data or not stats_history_data['data']:
        return df_battery, all_faults_list

    # El endpoint devuelve datos por vehículo, asumimos que solo pedimos uno
    vehicle_data = stats_history_data['data'][0]

    # 1. Procesar Historial de Batería
    if 'batteryMilliVolts' in vehicle_data:
        bat_history = vehicle_data['batteryMilliVolts']
        if bat_history:
            # (v38) pd.to_datetime() de un string con 'Z' es tz-aware
            df_battery_rows = [
                {'timestamp': pd.to_datetime(d['time']), 'value': float(d['value']) / 1000.0}
                for d in bat_history if d.get('value') is not None
            ]
            if df_battery_rows:
                df_battery = pd.DataFrame(df_battery_rows).set_index('timestamp').sort_index()
                
                # --- (v38) CORRECCIÓN CRÍTICA ---
                # Los timestamps ya son tz-aware (UTC), así que convertimos directamente
                if not df_battery.index.tz:
                     df_battery = df_battery.tz_localize(pytz.utc).tz_convert(MEXICO_TZ)
                else:
                     df_battery = df_battery.tz_convert(MEXICO_TZ)
                # ---------------------------------
                
                # Renombrar columna para que render_mini_chart funcione
                df_battery = df_battery.rename(columns={'value': 'value'})

    # 2. Procesar Historial de Fallas
    if 'faultCodes' in vehicle_data:
        all_faults_list = vehicle_data['faultCodes'] # Esta es ahora una LISTA de objetos de falla

    return df_battery, all_faults_list


# --- 4. FUNCIONES DE RENDERIZADO (CACHEADAS) ---

def render_vehicle_info_and_sensors(vehicle_obj, sensor_config):
    if not vehicle_obj:
        st.sidebar.error("No se ha seleccionado ningún vehículo.")
        return
        
    st.sidebar.markdown(f"<div class='vehicle-info-header'>Vehículo</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='vehicle-info-value'>{vehicle_obj.get('name', 'N/A')}</div>", unsafe_allow_html=True)
    
    if vehicle_obj.get('serial'):
        st.sidebar.markdown(f"<div class='vehicle-info-header' style='margin-top: 5px;'>Serial</div>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<div class='vehicle-info-value'>{vehicle_obj.get('serial')}</div>", unsafe_allow_html=True)
    
    st.sidebar.markdown(f"<div class='vehicle-info-header' style='margin-top: 5px;'>ID</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='vehicle-info-value'>{vehicle_obj.get('id')}</div>", unsafe_allow_html=True)
    
    st.sidebar.markdown(f"<div class='sensor-list-header' style='margin-top: 10px;'>Dispositivos Pareados</div>", unsafe_allow_html=True)
    
    if not sensor_config:
        st.sidebar.caption("No hay 'sensorConfiguration' para este vehículo.")
        st.sidebar.markdown("<div class='sensor-separator'></div>", unsafe_allow_html=True)
        return

    sensors = {}
    for area in sensor_config.get('areas', []):
        for s in area.get('temperatureSensors', []):
            if s['id'] not in sensors: sensors[s['id']] = {'name': s['name'], 'type': '🌡️ Temp'}
        for s in area.get('humiditySensors', []):
            if s['id'] not in sensors: 
                sensors[s['id']] = {'name': s['name'], 'type': '💧 Humedad'}
            else:
                sensors[s['id']]['type'] = '🌡️💧 Temp/Hum'
                
    for door in sensor_config.get('doors', []):
        s = door.get('sensor')
        if s and s['id'] not in sensors:
             sensors[s['id']] = {'name': s['name'], 'type': '🚪 Puerta'}

    if not sensors:
        st.sidebar.caption("No hay sensores EM/DM pareados.")
    else:
        for id, data in sensors.items():
            model = "EM21/EM22" if 'Temp' in data['type'] else "DM11"
            st.sidebar.markdown(f"<span class='sensor-list-item'>({model}) {data['name']}<br>&nbsp;&nbsp;↳ {id}</span>", unsafe_allow_html=True)
    
    st.sidebar.markdown("<div class='sensor-separator'></div>", unsafe_allow_html=True)


def parse_fault_code(fault_time, fault_obj, code_type):
    """(v35) Helper para parsear los diferentes tipos de códigos de falla."""
    codes_found = []
    
    if code_type == "obdii":
        if not fault_obj or 'diagnosticTroubleCodes' not in fault_obj:
            return []
        
        for module in fault_obj.get('diagnosticTroubleCodes', []):
            for dtc in module.get('confirmedDtcs', []):
                codes_found.append({
                    "Hora": fault_time, "Tipo": "OBD-II Confirmado",
                    "Codigo": dtc.get('dtcShortCode', 'N/A'),
                    "Desc": dtc.get('dtcDescription', 'N/A')
                })
            for dtc in module.get('pendingDtcs', []):
                codes_found.append({
                    "Hora": fault_time, "Tipo": "OBD-II Pendiente",
                    "Codigo": dtc.get('dtcShortCode', 'N/A'),
                    "Desc": dtc.get('dtcDescription', 'N/A')
                })
            for dtc in module.get('permanentDtcs', []):
                codes_found.append({
                    "Hora": fault_time, "Tipo": "OBD-II Permanente",
                    "Codigo": dtc.get('dtcShortCode', 'N/A'),
                    "Desc": dtc.get('dtcDescription', 'N/A')
                })
    
    elif code_type == "j1939":
        if not fault_obj or 'diagnosticTroubleCodes' not in fault_obj:
            return []
            
        for dtc in fault_obj.get('diagnosticTroubleCodes', []):
            codes_found.append({
                "Hora": fault_time, "Tipo": "J1939",
                "Codigo": f"SPN {dtc.get('spnId', 'N/A')} FMI {dtc.get('fmiId', 'N/A')}",
                "Desc": dtc.get('spnDescription', 'N/A')
            })
            
    return codes_found

def render_fault_codes(fault_codes_history_list, live_fault_codes_obj):
    """(v37) Actualizado para tomar lista de historial Y objeto en vivo."""
    st.sidebar.markdown(f"<div class='sensor-list-header'>🛠️ Códigos de Falla (Últimas 24h)</div>", unsafe_allow_html=True)
    
    # 1. Determinar el estado del Check Engine Light (del objeto en vivo)
    check_engine_light = False
    if live_fault_codes_obj and live_fault_codes_obj.get('obdii'):
        check_engine_light = live_fault_codes_obj['obdii'].get('checkEngineLightIsOn', False)
        
    if check_engine_light:
        st.sidebar.error("Check Engine Light: ENCENDIDA")
    else:
        st.sidebar.success("Check Engine Light: Apagada")

    # 2. Procesar la lista de historial de fallas
    if not fault_codes_history_list:
        st.sidebar.caption("No hay historial de códigos de falla en las últimas 24h.")
        st.sidebar.markdown("<div class='sensor-separator'></div>", unsafe_allow_html=True)
        return

    fault_list_for_df = []
    # (v38) Invertir la lista para mostrar los más recientes primero
    for fault_event in reversed(fault_codes_history_list):
        fault_time = pd.to_datetime(fault_event.get('time')).tz_convert(MEXICO_TZ).strftime('%H:%M')
        
        # Parsear códigos OBD-II
        obdii_codes = parse_fault_code(fault_time, fault_event.get('obdii'), "obdii")
        fault_list_for_df.extend(obdii_codes)
        
        # Parsear códigos J1939
        j1939_codes = parse_fault_code(fault_time, fault_event.get('j1939'), "j1939")
        fault_list_for_df.extend(j1939_codes)

    if fault_list_for_df:
        df_faults = pd.DataFrame(fault_list_for_df)
        st.sidebar.dataframe(df_faults, use_container_width=True, hide_index=True)
    else:
        st.sidebar.caption("No se encontraron códigos de falla activos en las últimas 24h.")
        
    st.sidebar.markdown("<div class='sensor-separator'></div>", unsafe_allow_html=True)


@st.cache_data
def render_mini_chart(series_data, color):
    if series_data.empty or series_data.isna().all():
        return st.caption("No hay datos históricos.")
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series_data.index, y=series_data,
        mode='lines', fill='tozeroy',
        line=dict(color=color, width=2),
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)'
    ))
    fig.update_layout(
        template="plotly_dark",
        height=70,
        margin=dict(t=5, b=5, l=5, r=5),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_door_status(door_series_1h):
    """(v39) Añadida lógica para mostrar el último evento *opuesto*."""
    
    if door_series_1h is None or door_series_1h.empty:
        status_str, status_time_str, status_class = "N/A", "Ping: N/A", "door-status-closed"
        prev_event_str = ""
    else:
        # 1. Obtener estado actual
        latest_status = door_series_1h.iloc[-1]
        latest_time = door_series_1h.index[-1]
        
        status_str = "CERRADA" if latest_status == 1 else "ABIERTA"
        status_class = "door-status-closed" if latest_status == 1 else "door-status-open"
        status_time_str = f"Último Ping: {latest_time.strftime('%H:%M:%S')}"
        
        # 2. Buscar último evento opuesto
        prev_event_str = ""
        try:
            # Encontrar todos los eventos *diferentes* al actual
            different_events = door_series_1h[door_series_1h != latest_status]
            if not different_events.empty:
                # Tomar el último de ellos (el más reciente)
                last_diff_time = different_events.index[-1]
                last_diff_status_val = different_events.iloc[-1]
                last_diff_status_str = "CERRADA" if last_diff_status_val == 1 else "ABIERTA"
                prev_event_str = f"(Previo: {last_diff_status_str} a las {last_diff_time.strftime('%H:%M:%S')})"
        except Exception as e:
            print(f"Error al buscar evento previo de puerta: {e}")
            prev_event_str = "" # Fallback
    
    st.markdown(f"<h5>🚪 Estado de Puerta</h5>", unsafe_allow_html=True)
    
    # (v39) HTML actualizado con .door-prev-event
    st.markdown(f"""
    <div class='door-status-box {status_class}'>
        <div>
            <div class='door-status-text'>{status_str}</div>
            <div class='door-ping-time'>{status_time_str}</div>
            <div class='door-prev-event'>{prev_event_str}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def render_history_charts(df, title_suffix):
    temp_chart_placeholder = st.empty()
    hum_chart_placeholder = st.empty()

    if not df.empty and 'temperature' in df.columns and df['temperature'].notna().any():
        fig_temp = px.line(df, y='temperature', labels={'temperature': 'Temperatura'})
        fig_temp.update_layout(
            title=f"Temperatura {title_suffix}",
            template="plotly_dark", height=200,
            xaxis_title="Hora (24h)", 
            xaxis_tickformat='%H:%M',
            yaxis_title="Temp °C",
            margin=dict(t=40, b=40, l=0, r=0),
            hovermode="x unified"
        )
        temp_chart_placeholder.plotly_chart(fig_temp, use_container_width=True, config={'displayModeBar': False})
    else:
        temp_chart_placeholder.info(f"No hay datos de temperatura disponibles ({title_suffix}).")

    if not df.empty and 'humidity' in df.columns and df['humidity'].notna().any():
        fig_hum = px.line(df, y='humidity', labels={'humidity': 'Humedad'})
        fig_hum.update_layout(
            title=f"Humedad {title_suffix}",
            template="plotly_dark", height=200,
            xaxis_title="Hora (24h)", 
            xaxis_tickformat='%H:%M',
            yaxis_title="Humedad %",
            margin=dict(t=40, b=40, l=0, r=0),
            hovermode="x unified"
        )
        hum_chart_placeholder.plotly_chart(fig_hum, use_container_width=True, config={'displayModeBar': False})
    else:
        hum_chart_placeholder.info(f"No hay datos de humedad disponibles ({title_suffix}).")
    
    return df

def render_webhook_log():
    """(v39) Esta función no cambia. Sigue leyendo del archivo,
    que ahora es poblado por `check_and_log_alerts`."""
    st.subheader("🔔 Últimas Alertas del Activo")
    log_file = utils.WEBHOOK_LOG_FILE
    
    if not os.path.exists(log_file):
        st.info("No se han recibido alertas del activo.")
        return

    alerts = []
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            for line in lines[-10:]:
                if line.strip():
                    alerts.append(json.loads(line))
    except Exception as e:
        st.error(f"Error al leer el registro de alertas: {e}")
        return
        
    alerts.reverse()
    
    if not alerts:
        st.info("No se han recibido alertas del activo.")
        return
        
    last_alert_time = alerts[0]['timestamp']
    if st.session_state.last_webhook_timestamp != last_alert_time:
        st.success("🔔 Nueva Alerta de Activo!", icon="🔔")
        st.session_state.last_webhook_timestamp = last_alert_time
        
    for alert in alerts:
        vehicle_name = alert.get('vehicle_name', 'N/A')
        driver_name = alert.get('driver_name', '')
        context = f"Vehículo: {vehicle_name}"
        if driver_name and driver_name != 'N/A':
            context += f" | Conductor: {driver_name}"
            
        st.markdown(f"""
        <div class='webhook-alert'>
            <div class='webhook-alert-header'>
                <strong>{alert['type']}</strong>
                <span style='float: right;'>{alert['timestamp']}</span>
            </div>
            <div class='webhook-alert-message'>{alert['message']}</div>
            <div class='webhook-alert-header' style='margin-top: 5px;'>{context}</div>
        </div>
        """, unsafe_allow_html=True)

@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def generate_map_data(live_stats, vehicle_name):
    """Usa el 'live_stats' (snapshot) para el mapa,
       para obtener la ubicación más reciente."""
    if live_stats and live_stats.get('gps'):
        gps_data = live_stats['gps']
        lat, lon = gps_data.get('latitude'), gps_data.get('longitude')
        if lat and lon:
            map_center = [lat, lon]
            gps_time_str = pd.to_datetime(gps_data.get('time')).tz_convert(MEXICO_TZ).strftime('%H:%M:%S')
            popup_html = f"{vehicle_name}<br>Vel: {gps_data.get('speedMilesPerHour', 0) * 1.609:.1f} km/h<br>Hora: {gps_time_str}"
            return map_center, popup_html
    return [20.6736, -103.344], "No hay datos de GPS."

@st.cache_data
def get_ia_prediction(_ai_model, temperature_series, step_sec_ai):
    if not temperature_series.empty and len(temperature_series) >= 5:
        return _ai_model.get_temperature_forecast(
            temperature_series,
            steps_ahead=12,
            step_seconds=step_sec_ai
        )
    return None, []

# --- (v39) NUEVO MOTOR DE ALERTAS (Reemplaza Webhooks) ---
def check_and_log_alerts(current_stats, previous_stats, vehicle_name):
    """Compara el estado actual con el previo y registra alertas."""
    if not previous_stats or not current_stats:
        print("Motor de Alertas: Esperando al siguiente ciclo (no hay datos previos).")
        return # No alertar en la primera ejecución

    current_time_str = datetime.now(MEXICO_TZ).strftime('%Y-%m-%d %H:%M:%S')

    # 1. Alerta de Check Engine Light
    try:
        current_cel = current_stats.get('faultCodes', {}).get('obdii', {}).get('checkEngineLightIsOn', False)
        prev_cel = previous_stats.get('faultCodes', {}).get('obdii', {}).get('checkEngineLightIsOn', False)
        
        if current_cel and not prev_cel:
            alert_info = {
                "type": "Alerta de Motor",
                "driver_name": "N/A",
                "vehicle_name": vehicle_name,
                "message": "Luz de Check Engine (MIL) se ha ENCENDIDO.",
                "timestamp": current_time_str
            }
            utils.log_alert(alert_info)
            print("ALERTA GENERADA: Check Engine Light ON")
            
    except Exception as e:
        print(f"Error al procesar alerta de Check Engine: {e}")

    # (Futuro) Aquí se pueden añadir más comprobaciones:
    # Ej. if current_temp > 50 and prev_temp <= 50: ...
    # Ej. if current_door == 'ABIERTA' for > 10 min: ...


# --- 5. INICIALIZACIÓN DE LA APP ---

vehicle_names, vehicle_map_obj = load_vehicle_list(st.session_state.api_client)

if not vehicle_names:
    st.error("No se pudieron cargar los vehículos. ¿La clave de API es correcta y tiene permisos?")
    st.stop()

# --- 6. BARRA LATERAL (SIDEBAR) ---

st.sidebar.title("Panel de Control")

def on_vehicle_change():
    st.session_state.selected_vehicle_obj = vehicle_map_obj.get(st.session_state.selected_vehicle_name)
    if st.session_state.selected_vehicle_obj:
        st.session_state.sensor_config = st.session_state.selected_vehicle_obj.get('sensorConfiguration')
    else:
        st.session_state.sensor_config = None
    # Limpiar caché de datos
    st.cache_data.clear()
    # (v39) Reiniciar el estado de alertas
    st.session_state.previous_live_stats = None


if st.session_state.selected_vehicle_name is None and vehicle_names:
    st.session_state.selected_vehicle_name = vehicle_names[0]
    on_vehicle_change()

st.sidebar.selectbox(
    "Selecciona un Vehículo:",
    vehicle_names,
    key='selected_vehicle_name',
    on_change=on_vehicle_change,
    label_visibility="collapsed"
)

if not st.session_state.selected_vehicle_obj:
    st.error("Error al cargar datos del vehículo.")
    st.stop()

# --- 7. CARGA DE DATOS PRINCIPAL (CACHEADA) ---
selected_vehicle_id = st.session_state.selected_vehicle_obj['id']
selected_vehicle_name = st.session_state.selected_vehicle_obj['name']
sensor_config = st.session_state.sensor_config

# 1. SNAPSHOT (Último valor de KPIs y GPS)
live_stats, live_temp_snapshot, live_hum_snapshot = fetch_live_kpis(
    st.session_state.api_client, sensor_config, selected_vehicle_id
)
live_fault_codes_obj = live_stats.get('faultCodes') if live_stats else None


# 2. HISTORIAL DE SENSORES (1h, 30s) -> Para KPIs de Puerta/Temp/Hum
history_results_1h, column_map_1h = fetch_live_sensor_history(
    st.session_state.api_client, sensor_config, 60, 30
)
df_history_1h = process_sensor_history_data(
    history_results_1h, column_map_1h, step_seconds=30
)

# 3. HISTORIAL DE SENSORES (24h, 10min) -> Para Gráficos de Tendencia
history_results_24h, column_map_24h = fetch_live_sensor_history(
    st.session_state.api_client, sensor_config, 1440, 600
)
df_history_24h = process_sensor_history_data(
    history_results_24h, column_map_24h, step_seconds=600
)

# 4. (v37) HISTORIAL DE VEHÍCULO (24h, 10min) -> Para Gráfico de Batería y Lista de Fallas
raw_stats_history = fetch_vehicle_stats_history(
    st.session_state.api_client, selected_vehicle_id, 1440
)
df_battery_history_24h, all_fault_codes_list = process_vehicle_stats_history(raw_stats_history)

# --- (v39) EJECUTAR MOTOR DE ALERTAS ---
check_and_log_alerts(live_stats, st.session_state.previous_live_stats, selected_vehicle_name)
st.session_state.previous_live_stats = live_stats # Guardar estado actual para el próximo ciclo


# --- 8. RENDERIZADO DE LA BARRA LATERAL ---
render_vehicle_info_and_sensors(st.session_state.selected_vehicle_obj, sensor_config)
# (v37) Pasar ambos: la lista de historial y el objeto en vivo
render_fault_codes(all_fault_codes_list, live_fault_codes_obj)


# --- 9. RENDERIZADO DEL CUERPO PRINCIPAL ---

# --- FILA DE KPIs (ACTUALIZADA CADA 30S) ---
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    # (v36) Pasando la serie de historial de alta frecuencia (1h)
    render_door_status(df_history_1h.get('doorClosed'))

with kpi_col2:
    with st.container(border=True, height=185):
        # (v39) KPI de Temperatura basado en el historial de 1h
        temp_val_str, temp_time_str = "N/A", "(N/A)"
        if not df_history_1h.empty and 'temperature' in df_history_1h.columns:
            latest_temp = df_history_1h['temperature'].iloc[-1]
            latest_temp_time = df_history_1h.index[-1]
            temp_val_str = f"{latest_temp:.1f} °C"
            temp_time_str = latest_temp_time.strftime('(%H:%M)')
        
        st.metric(label=f"🌡️ Temperatura {temp_time_str}", value=temp_val_str)
        
        # (v36) Mini-gráfico usa historial de alta frecuencia (1h)
        if 'temperature' in df_history_1h.columns:
            render_mini_chart(df_history_1h['temperature'], "#FFA500")
        else:
            st.caption("No hay historial (1h).")

with kpi_col3:
    with st.container(border=True, height=185):
        # (v39) KPI de Humedad basado en el historial de 1h
        hum_val_str, hum_time_str = "N/A", "(N/A)"
        if not df_history_1h.empty and 'humidity' in df_history_1h.columns:
            latest_hum = df_history_1h['humidity'].iloc[-1]
            latest_hum_time = df_history_1h.index[-1]
            hum_val_str = f"{latest_hum:.0f} %"
            hum_time_str = latest_hum_time.strftime('(%H:%M)')
        
        st.metric(label=f"💧 Humedad {hum_time_str}", value=hum_val_str)

        # (v36) Mini-gráfico usa historial de alta frecuencia (1h)
        if 'humidity' in df_history_1h.columns:
            render_mini_chart(df_history_1h['humidity'], "#3498DB")
        else:
            st.caption("No hay historial (1h).")
            
with kpi_col4:
    with st.container(border=True, height=185):
        # (v39) KPI de Batería basado en el historial de 24h
        bat_val, bat_time_str = "N/A", "(N/A)"
        if not df_battery_history_24h.empty and 'value' in df_battery_history_24h.columns:
            latest_bat = df_battery_history_24h['value'].iloc[-1]
            latest_bat_time = df_battery_history_24h.index[-1]
            bat_val = f"{latest_bat:.2f} V"
            bat_time_str = latest_bat_time.strftime('(%H:%M)')
        
        st.metric(label=f"🔋 Batería {bat_time_str}", value=bat_val)

        # (v37) Renderizar mini-gráfico desde el historial de 24h
        if 'value' in df_battery_history_24h.columns and len(df_battery_history_24h['value']) > 1:
            render_mini_chart(df_battery_history_24h['value'], "#2ECC71") # Verde
        else:
            st.caption("No hay historial de batería (24h).")

st.markdown("---") # Separador

# --- FILA PRINCIPAL (MAPA Y GRÁFICOS) ---
col1, col2 = st.columns((6, 4)) 

with col1:
    # --- MAPA (Usa Snapshot) ---
    st.subheader("📍 Ubicación en Tiempo Real")
    
    map_center, popup_html = generate_map_data(live_stats, selected_vehicle_name)

    m = folium.Map(location=map_center, zoom_start=14, tiles="CartoDB dark_matter")
    if popup_html != "No hay datos de GPS.":
        icon_html = """
        <div style="width: 20px; height: 20px; background-color: #2ECC71; border-radius: 50%;
            border: 2px solid #FFFFFF; box-shadow: 0 0 10px #2ECC71, 0 0 12px #2ECC71;
            animation: pulse 1.5s infinite;"></div>
        <style>@keyframes pulse { 0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 15px rgba(46, 204, 113, 0); }
            100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); } }
        </style>
        """
        folium.Marker(location=map_center, popup=popup_html, icon=DivIcon(html=icon_html)).add_to(m)
    
    st_folium(m, width=None, height=500, use_container_width=True, returned_objects=[])
    
    # --- LOG DE WEBHOOKS (AHORA POBLADO POR EL MOTOR DE ALERTAS) ---
    st.markdown("---")
    render_webhook_log() 

with col2:
    # --- GRÁFICOS DE 24H (Cacheados) ---
    st.subheader(f"📈 Tendencias (24h)")
    
    df_ai_input = pd.DataFrame()
    step_sec_ai = 600
    
    # (v36) Los gráficos principales usan el historial de 24h
    df_display = render_history_charts(df_history_24h, "(Últimas 24h)")
    
    if 'temperature' in df_display.columns:
        df_ai_input = df_display.dropna(subset=['temperature'])[['temperature']]
        
    st.markdown("---")

    # --- ANALÍTICA AVANZADA (Cacheada) ---
    st.subheader("🔮 Predicción de Temperatura (IA)")
    with st.container(border=True):
        # (v36) La IA usa el historial de 24h
        forecast_index, forecast_values = get_ia_prediction(
            st.session_state.ai_model, 
            df_ai_input['temperature'], 
            step_sec_ai
        )
        
        if forecast_values is not None and len(forecast_values) > 0:
            df_forecast = pd.DataFrame({'timestamp': forecast_index, 'Predicción': forecast_values}).set_index('timestamp')
            df_real = df_ai_input[['temperature']].rename(columns={'temperature': 'Real'})
            df_plot = pd.concat([df_real, df_forecast])
            
            fig_fc = px.line(df_plot, markers=True, title="Predicción de Temperatura")
            fig_fc.update_layout(template="plotly_dark", yaxis_title="Temperatura °C", margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig_fc, use_container_width=True, config={'displayModeBar': False})
            st.caption(f"Nota: El modelo LSTM se entrena sobre la marcha con los datos seleccionados.")
        else:
            st.info(f"Se necesitan al menos 5 puntos de datos para la predicción (actuales: {len(df_ai_input)}).")
