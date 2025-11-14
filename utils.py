# --------------------------------------------------------------------------
# utils.py
#
# Lógica de Backend, Cliente de API de Samsara e IA.
#
# v39 (Solución Profesional de Webhooks)
# - ELIMINADO (CRÍTICO): Se eliminó TODO el código relacionado con
#   Flask, threading, hmac, hashlib. Las funciones `run_flask_app`,
#   `start_webhook_thread` y `handle_samsara_webhook` han sido
#   borradas. Esto soluciona los errores 303 y "Address in use"
#   en Streamlit Cloud.
# - NUEVO: Se añadió una función simple `log_alert(alert_info)`
#   que simplemente escribe en el archivo `webhook_log.jsonl`.
#   Esta función es llamada por el nuevo motor de alertas en `app.py`.
# - MANTENIDO (v37): Se mantiene la función `get_vehicle_stats_history`
#   para obtener el historial de batería y fallas.
# --------------------------------------------------------------------------

import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import os
import time
from datetime import datetime, timedelta, timezone
import pytz
import json
from dotenv import load_dotenv

# (v39) Imports de Flask/threading eliminados

# --- 1. CONFIGURACIÓN INICIAL Y VARIABLES DE ENTORNO ---

load_dotenv()
SAMSARA_API_KEY = os.getenv("SAMSARA_API_KEY")
SAMSARA_WEBHOOK_SECRET = os.getenv("SAMSARA_WEBHOOK_SECRET") # (v39) Ya no se usa para hmac, pero se mantiene por si acaso

if not SAMSARA_API_KEY:
    raise ValueError("La variable SAMSARA_API_KEY no está configurada en el archivo .env")
# (v39) El secret ya no es crítico para el funcionamiento
# if not SAMSARA_WEBHOOK_SECRET:
#     raise ValueError("La variable SAMSARA_WEBHOOK_SECRET no está configurada en el archivo .env")

SAMSARA_API_URL = "https://api.samsara.com"
MEXICO_TZ = pytz.timezone("America/Mexico_City") 
WEBHOOK_LOG_FILE = "webhook_log.jsonl"
# (v39) Puerto de Webhook eliminado
# WEBHOOK_PORT = 5001

# --- 2. CLIENTE DE LA API DE SAMSARA ---

class SamsaraAPIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.v1_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Api-Key": self.api_key
        }
        print(f"Cliente de API inicializado.")

    def _make_request(self, endpoint, method="GET", params=None, json_data=None, is_v1=False):
        """Helper function to make API requests with error handling."""
        url = f"{SAMSARA_API_URL}{endpoint}"
        headers = self.v1_headers if is_v1 else self.headers
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=headers, params=params, json=json_data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"Error HTTP: {http_err} - {response.text}")
        except requests.exceptions.RequestException as req_err:
            print(f"Error de Petición: {req_err}")
        except Exception as e:
            print(f"Error inesperado en la petición de API: {e}")
        return None

    def get_vehicles(self):
        """Obtiene la lista de todos los vehículos, incluyendo su 'sensorConfiguration'."""
        print("API: Obteniendo lista de vehículos (con sensorConfiguration)...")
        endpoint = "/fleet/vehicles"
        params = {'limit': 500}
        data = self._make_request(endpoint, method="GET", params=params)
        if data and 'data' in data:
            print(f"API: Encontrados {len(data['data'])} vehículos activos.")
            return data['data']
        print("API: No se encontraron vehículos.")
        return []

    def get_live_stats(self, vehicle_id):
        """(SNAPSHOT) Obtiene GPS, Batería y Fallas (último valor)."""
        print(f"API-LIVE: Obteniendo estadísticas (GPS, Batería, Fallas) para {vehicle_id}...")
        
        endpoint = "/fleet/vehicles/stats" 
        params = {
            'vehicleIds': str(vehicle_id),
            'types': 'gps,batteryMilliVolts,faultCodes'
        }
        data = self._make_request(endpoint, method="GET", params=params)
        if data and 'data' in data and len(data['data']) > 0:
            # Devuelve el objeto de estadísticas completo
            return data['data'][0]
        print(f"API-LIVE: No se encontraron estadísticas para el vehículo {vehicle_id}.")
        return None

    def _get_sensor_ids_from_config(self, sensor_configuration):
        """Helper para extraer IDs de un objeto sensorConfiguration."""
        sensor_ids = set()
        
        if not sensor_configuration:
            return []
        
        if sensor_configuration.get('areas'):
            for area in sensor_configuration.get('areas', []):
                for temp_sensor in area.get('temperatureSensors', []):
                    sensor_ids.add(int(temp_sensor['id']))
                for hum_sensor in area.get('humiditySensors', []):
                    sensor_ids.add(int(hum_sensor['id']))
        
        if sensor_configuration.get('doors'):
            for door in sensor_configuration.get('doors', []):
                if door.get('sensor'):
                    sensor_ids.add(int(door['sensor']['id']))
        
        return list(sensor_ids)

    def get_live_sensor_kpis(self, sensor_configuration, vehicle_id):
        """(SNAPSHOT) Obtiene los datos de snapshot para Temp y Humedad."""
        sensor_ids = self._get_sensor_ids_from_config(sensor_configuration)
        
        if not sensor_ids:
            print(f"API-LIVE: No hay sensores configurados para el vehículo {vehicle_id}.")
            return [], []
            
        print(f"API-LIVE: Consultando KPIs para {len(sensor_ids)} sensores: {sensor_ids}")
        json_payload = {"sensors": sensor_ids}

        temp_data = self._make_request("/v1/sensors/temperature", method="POST", json_data=json_payload, is_v1=True)
        hum_data = self._make_request("/v1/sensors/humidity", method="POST", json_data=json_payload, is_v1=True)
        
        def filter_sensors(data, vehicle_id_str):
            if not data or 'sensors' not in data:
                return []
            return [s for s in data['sensors'] if str(s.get('vehicleId')) == vehicle_id_str]

        filtered_temp = filter_sensors(temp_data, str(vehicle_id))
        filtered_hum = filter_sensors(hum_data, str(vehicle_id))

        print(f"API-LIVE: Datos de KPI recibidos para {vehicle_id}: {len(filtered_temp)} T, {len(filtered_hum)} H.")
        # (v34) Devuelve solo temp y humedad
        return filtered_temp, filtered_hum

    def build_sensor_payload_from_config(self, sensor_configuration):
        """Construye el 'series_query' y el 'column_map' dinámicamente."""
        if not sensor_configuration:
            return [], {}
            
        sensor_map = {} 
        
        for area in sensor_configuration.get('areas', []):
            for temp_sensor in area.get('temperatureSensors', []):
                widget_id = str(temp_sensor['id'])
                if widget_id not in sensor_map:
                    sensor_map[widget_id] = []
                if "probeTemperature" not in sensor_map[widget_id]:
                     sensor_map[widget_id].append("ambientTemperature")
                
            for hum_sensor in area.get('humiditySensors', []):
                widget_id = str(hum_sensor['id'])
                if widget_id not in sensor_map:
                    sensor_map[widget_id] = []
                if "humidity" not in sensor_map[widget_id]:
                    sensor_map[widget_id].append("humidity")

        for door in sensor_configuration.get('doors', []):
            if door.get('sensor'):
                widget_id = str(door['sensor']['id'])
                if widget_id not in sensor_map:
                    sensor_map[widget_id] = []
                if "doorClosed" not in sensor_map[widget_id]:
                    sensor_map[widget_id].append("doorClosed")
        
        series_query = []
        column_map = {} 
        current_index = 0
        
        for widget_id_str, fields in sensor_map.items():
            widget_id_int = int(widget_id_str)
            for field in fields:
                series_query.append({
                    "widgetId": widget_id_int,
                    "field": field
                })
                
                if field == "ambientTemperature" or field == "probeTemperature":
                    column_map[current_index] = "temperature"
                elif field == "humidity":
                    column_map[current_index] = "humidity"
                elif field == "doorClosed":
                    column_map[current_index] = "doorClosed"
                
                current_index += 1
                
        return series_query, column_map


    def get_live_sensor_history(self, sensor_configuration, time_window_minutes=60, step_seconds=30):
        """(HISTORIAL) Obtiene historial de SENSORES (Temp, Hum, Puerta)."""
        print(f"API-HIST-SENSORES: Obteniendo historial (últimos {time_window_minutes} min, step {step_seconds}s)...")
        
        series_query, column_map = self.build_sensor_payload_from_config(sensor_configuration)
        
        if not series_query:
            print(f"API-HIST-SENSORES: No hay sensores configurados para el historial en vivo.")
            return [], {} 

        end_time_utc = datetime.now(pytz.utc)
        start_time_utc = end_time_utc - timedelta(minutes=time_window_minutes)
        step_ms = int(step_seconds * 1000)
        
        if len(series_query) > 40:
            print(f"API-HIST-SENSORES: Demasiadas series ({len(series_query)}), truncando a 40.")
            series_query = series_query[:40]
            new_column_map = {k: v for k, v in column_map.items() if k < 40}
            column_map = new_column_map

        json_payload = {
            "endMs": int(end_time_utc.timestamp() * 1000),
            "startMs": int(start_time_utc.timestamp() * 1000), 
            "stepMs": int(step_ms),
            "series": series_query,
            "fillMissing": "withNull" # (v34) Cambiado a withNull, ffill se hace en pandas
        }
        
        data = self._make_request("/v1/sensors/history", method="POST", json_data=json_payload, is_v1=True)
        
        if data and 'results' in data:
            print(f"API-HIST-SENSORES: Encontrados {len(data['results'])} puntos de datos históricos.")
            return data['results'], column_map 
            
        print("API-HIST-SENSORES: No se encontraron datos históricos.")
        return [], column_map

    def get_vehicle_stats_history(self, vehicle_id, time_window_minutes):
        """(v37) (HISTORIAL) Obtiene historial de Batería, GPS y Fallas."""
        print(f"API-HIST-VEHICULO: Obteniendo historial (últimos {time_window_minutes} min)...")
        
        end_time_utc = datetime.now(pytz.utc)
        start_time_utc = end_time_utc - timedelta(minutes=time_window_minutes)
        
        endpoint = "/fleet/vehicles/stats/history"
        params = {
            'vehicleIds': str(vehicle_id),
            'types': 'batteryMilliVolts,faultCodes,gps',
            'startTime': start_time_utc.isoformat(),
            'endTime': end_time_utc.isoformat()
        }
        
        data = self._make_request(endpoint, method="GET", params=params)
        
        if data and 'data' in data:
            print(f"API-HIST-VEHICULO: Historial de estadísticas de vehículo encontrado.")
            return data
        
        print("API-HIST-VEHICULO: No se encontraron datos históricos de estadísticas.")
        return None

# --- 3. MODELOS DE INTELIGENCIA ARTIFICIAL (IA) ---

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

class AIModels:
    def __init__(self):
        self.model = LSTMForecaster()
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def detect_anomalies(self, data_series):
        if data_series.empty or len(data_series) < 2: 
            return pd.DataFrame(columns=[data_series.name, 'zscore', 'timestamp'])
        df = pd.DataFrame(data_series.copy())
        df['zscore'] = zscore(df[data_series.name])
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        return df

    def get_temperature_forecast(self, data_series, steps_ahead=12, step_seconds=30):
        """(v29) step_seconds permite predicción correcta para 1h (30s) o 24h (600s)."""
        print("IA: Generando predicción LSTM...")
        if len(data_series) < 5:
            print("IA: No hay suficientes datos para predecir.")
            return None, []
        try:
            data = data_series.values.astype(float)
            data_normalized = self.scaler.fit_transform(data.reshape(-1, 1))
            data_normalized = torch.FloatTensor(data_normalized).view(-1)
            train_window = 4
            inout_seq = []
            for i in range(len(data_normalized) - train_window):
                train_seq = data_normalized[i:i+train_window]
                train_label = data_normalized[i+train_window:i+train_window+1]
                inout_seq.append((train_seq, train_label))
            if not inout_seq:
                print("IA: No se pudieron crear secuencias de entrenamiento.")
                return None, []
            
            self.model.train()
            for i in range(25): # Epochs
                for seq, labels in inout_seq:
                    self.optimizer.zero_grad()
                    self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                            torch.zeros(1, 1, self.model.hidden_layer_size))
                    y_pred = self.model(seq)
                    single_loss = self.loss_function(y_pred, labels)
                    single_loss.backward()
                    self.optimizer.step()
            
            self.model.eval()
            future_predictions = []
            test_inputs = data_normalized[-train_window:].tolist()
            for i in range(steps_ahead):
                seq = torch.FloatTensor(test_inputs[-train_window:])
                with torch.no_grad():
                    self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                            torch.zeros(1, 1, self.model.hidden_layer_size))
                    pred = self.model(seq)
                    future_predictions.append(pred.item())
                    test_inputs.append(pred.item())
            
            forecast_values = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
            
            last_timestamp = data_series.index[-1]
            time_delta = pd.Timedelta(seconds=step_seconds) 
            forecast_index = [last_timestamp + time_delta * i for i in range(1, steps_ahead + 1)]
            
            print("IA: Predicción generada exitosamente.")
            return forecast_index, forecast_values
        except Exception as e:
            print(f"Error durante la predicción LSTM: {e}")
            return None, []

# --- 4. (v39) NUEVO REGISTRADOR DE ALERTAS ---

def log_alert(alert_info):
    """
    (v39) Reemplaza la lógica del webhook. Simplemente escribe
    una alerta generada internamente en el archivo de registro.
    """
    try:
        with open(WEBHOOK_LOG_FILE, "a") as f: # 'a' = append
            f.write(json.dumps(alert_info) + "\n") # Escribir como JSON Line
        print(f"Alerta Interna Registrada: {alert_info['message']}")
    except Exception as e:
        print(f"Error al escribir en el archivo de registro de alertas: {e}")

# --- (v39) Lógica de Flask/Threading ELIMINADA ---
