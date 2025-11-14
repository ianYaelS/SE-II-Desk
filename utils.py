# --------------------------------------------------------------------------
# utils.py
#
# Lógica de Backend, Cliente de API de Samsara, IA y Servidor Webhook.
#
# v37 (Solución Profesional de Historial vs. Snapshot)
# - NUEVA FUNCIÓN: `get_vehicle_stats_history` para llamar a
#   `fleet/vehicles/stats/history?types=gps,batteryMilliVolts,faultCodes`.
#   Esto permite obtener la serie de tiempo de batería y fallas,
#   en lugar de solo el snapshot.
# - MANTENIDO (ESTABILIDAD): Se mantiene el `try/except OSError`
#   en `run_flask_app` para evitar el crash de "Address already in use"
#   en el puerto 5001.
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
from flask import Flask, request, abort
import threading
import hmac
import hashlib
import json
from dotenv import load_dotenv

# --- 1. CONFIGURACIÓN INICIAL Y VARIABLES DE ENTORNO ---

load_dotenv()
SAMSARA_API_KEY = os.getenv("SAMSARA_API_KEY")
SAMSARA_WEBHOOK_SECRET = os.getenv("SAMSARA_WEBHOOK_SECRET")

if not SAMSARA_API_KEY:
    raise ValueError("La variable SAMSARA_API_KEY no está configurada en el archivo .env")
if not SAMSARA_WEBHOOK_SECRET:
    raise ValueError("La variable SAMSARA_WEBHOOK_SECRET no está configurada en el archivo .env")

SAMSARA_API_URL = "https://api.samsara.com"
MEXICO_TZ = pytz.timezone("America/Mexico_City") 
WEBHOOK_LOG_FILE = "webhook_log.jsonl"
WEBHOOK_PORT = 5000

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

    def get_vehicle_stats_history(self, vehicle_id, time_window_minutes, step_seconds):
        """(v37 - NUEVO) (HISTORIAL) Obtiene historial de Batería, GPS y Fallas."""
        print(f"API-HIST-VEHICULO: Obteniendo historial (últimos {time_window_minutes} min, step {step_seconds}s)...")
        
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

# --- 4. SERVIDOR WEBHOOK (FLASK) ---
flask_app = Flask(__name__)

@flask_app.route('/webhook', methods=['POST'])
def handle_samsara_webhook():
    print("¡Webhook recibido!")
    signature = request.headers.get('X-Samsara-Signature')
    if not signature:
        print("Error de Webhook: Falta la cabecera X-Samsara-Signature.")
        abort(400, "Falta la firma.")
    try:
        sig_hash = signature.split('=')[-1]
        expected_hash = hmac.new(
            key=SAMSARA_WEBHOOK_SECRET.encode('utf-8'),
            msg=request.data,
            digestmod=hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig_hash, expected_hash):
            print("Error de Webhook: Firma no válida.")
            abort(403, "Firma no válida.")
    except Exception as e:
        print(f"Error al verificar la firma: {e}")
        abort(500, "Error de verificación.")

    print("Firma de Webhook verificada con éxito.")
    
    try:
        data = request.json
        alert_info = {}
        
        current_time_str = datetime.now(MEXICO_TZ).strftime('%Y-%m-%d %H:%M:%S')

        if data.get('webhookType') == 'form':
            form_data = data.get('data')
            if form_data:
                answer_value = form_data.get('formAnswer', {}).get('answerValue')
                if answer_value:
                    alert_info = {
                        "type": f"Formulario: {form_data.get('form', {}).get('name', 'N/A')}",
                        "driver_name": form_data.get('driver', {}).get('name', 'N/A'),
                        "vehicle_name": form_data.get('vehicle', {}).get('name', 'N/A'),
                        "message": f"{answer_value}",
                        "timestamp": current_time_str
                    }
        
        elif data.get('webhookType') == 'alert':
             alert_data = data.get('data')
             if alert_data:
                 alert_type = alert_data.get('type', 'Alerta General')
                 driver_name = alert_data.get('driver', {}).get('name', 'N/A')
                 vehicle_name = alert_data.get('vehicle', {}).get('name', 'N/A')
                 
                 if vehicle_name == 'N/A' and alert_data.get('tags'):
                     for tag in alert_data.get('tags', []):
                         if tag.get('parentTagId') is None:
                             vehicle_name = tag.get('name', 'N/A')
                             break
                             
                 alert_info = {
                        "type": f"{alert_type} ({alert_data.get('severity', 'info')})",
                        "driver_name": driver_name,
                        "vehicle_name": vehicle_name,
                        "message": alert_data.get('description', 'Alerta sin descripción'),
                        "timestamp": current_time_str
                 }

        if alert_info:
            try:
                with open(WEBHOOK_LOG_FILE, "a") as f:
                    f.write(json.dumps(alert_info) + "\n")
                print(f"Alerta de Webhook registrada: {alert_info}")
            except Exception as e:
                print(f"Error al escribir en el archivo de registro webhook: {e}")

        return "OK", 200
    except Exception as e:
        print(f"Error al procesar el cuerpo del JSON del webhook: {e}")
        abort(400, "JSON malformado.")

def run_flask_app():
    """(v36) Añadido try/except para OSError para manejar 'Address already in use'."""
    try:
        # (v37) Desactivar el logger de "Werkzeug" para una consola más limpia
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        flask_app.run(host='0.0.0.0', port=WEBHOOK_PORT)
    except OSError as e:
        if e.errno == 98 or e.errno == 48: # 98 (Linux) y 48 (macOS) son "Address already in use"
            print(f"ADVERTENCIA: El puerto {WEBHOOK_PORT} ya está en uso. Es probable que otro hilo ya lo esté escuchando.")
        else:
            print(f"Error inesperado de OSError en el hilo de Flask: {e}")
    except Exception as e:
        print(f"Error inesperado en el hilo de Flask: {e}")

def start_webhook_thread():
    print("Iniciando hilo del servidor webhook...")
    webhook_thread = threading.Thread(target=run_flask_app, daemon=True)
    webhook_thread.start()
    print(f"Hilo del servidor webhook iniciado. Escuchando en el puerto {WEBHOOK_PORT}.")