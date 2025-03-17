from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta, timezone
import pickle
from cachetools import TTLCache
import asyncio
import httpx
import os
import tracemalloc
from tensorflow.keras import backend as K

# Start real-time geheugen monitoring
tracemalloc.start()

# initialize server
app = FastAPI()

# initialize cache (TTL van 1200 seconden en max 2000 sessies)
aggregated_data = TTLCache(maxsize=2000, ttl=1200)

# Importeer tokenizer en model
with open('tokenizer-test.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = tf.keras.models.load_model('LSTM-test.keras')

# Configureer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.sanitairwinkel.nl/"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-Apikey", "Content-Type"],
)

# NIEUW: Gecentraliseerde HTTP-client om geheugen te besparen
CLIENT = httpx.AsyncClient()

@app.on_event("shutdown")
async def close_http_client():
    await CLIENT.aclose()

# NIEUW: Houdt achtergrondtaken bij en ruimt ze op na voltooiing
background_tasks = set()

def milliseconds_to_datetime(ms):
    return datetime.fromtimestamp(ms / 1000.0, timezone.utc)

def clean_expired_sessions():
    """Verwijdert verlopen sessies uit de cache om geheugen te besparen."""
    current_time = datetime.now(timezone.utc)
    for session in list(aggregated_data.keys()):
        session_start_time = aggregated_data[session]['timestamps'][0]
        if (current_time - session_start_time).total_seconds() > 1200:
            del aggregated_data[session]

async def send_prediction(prediction: dict):
    """Stuur een asynchrone API-aanvraag zonder geheugen te blokkeren."""
    url = "https://api.rorix.nl/recommendation"
    headers = {
        "X-Apikey": "fPhVEkGpS7o3kQc41nwqdWE7VZYx6ZvY",
        "Content-Type": "application/json"
    }
    try:
        # NIEUW: timeout toegevoegd om hangende requests te voorkomen
        response = await CLIENT.post(url, headers=headers, json=prediction, timeout=5.0)
        if response.status_code == 200:
            print(response.json())
        elif response.status_code == 401:
            print("AUTHENTICATIE FOUT: 401")
        else:
            print(f"API-ERROR {response.status_code}: {response.text}")
    except httpx.ConnectTimeout:
        print("FOUT: API reageerde niet op tijd.")
    except Exception as e:
        print(f"ONVERWACHTE FOUT IN send_prediction(): {str(e)}")

# Functie om events te verwerken en voorspellingen te doen
async def process_event(event):
    """Verwerkt een event en voorspelt of een gebruiker een pop-up moet krijgen."""
    
    # NIEUW: Opschonen van verlopen sessies
    clean_expired_sessions()
    
    session_id = event['ga_stream_cookie']
    event_timestamp = milliseconds_to_datetime(int(event['timestamp']))
    
    if session_id not in aggregated_data:
        aggregated_data[session_id] = {
            'events': [event['event']],
            'timestamps': [event_timestamp],
            'time_since_session_start': [timedelta(seconds=0)],
            'time_between_events': [timedelta(seconds=0)],
            'number_of_events': 1,
            'prediction': 0
        }
    else:
        session_start_time = aggregated_data[session_id]['timestamps'][0]
        last_event_time = aggregated_data[session_id]['timestamps'][-1]
        aggregated_data[session_id]['events'].append(event['event'])
        aggregated_data[session_id]['timestamps'].append(event_timestamp)
        aggregated_data[session_id]['time_since_session_start'].append(event_timestamp - session_start_time)
        aggregated_data[session_id]['time_between_events'].append(event_timestamp - last_event_time)
        aggregated_data[session_id]['number_of_events'] += 1

    if aggregated_data[session_id]['number_of_events'] >= 4 and aggregated_data[session_id]['prediction'] == 0:
        # Tokenize de events naar gehele getallen
        tokenized_events = [tokenizer.texts_to_sequences([x])[0][0] for x in aggregated_data[session_id]['events']]
        tsst_list = [int(t.total_seconds()) for t in aggregated_data[session_id]['time_since_session_start']]
        tbe_list = [int(t.total_seconds()) for t in aggregated_data[session_id]['time_between_events']]

        # Pre-pad de lijsten tot lengte 50 met -1 als fill-value
        tokenized_events = list(pad_sequences([tokenized_events], maxlen=50, value=-1)[0])
        tsst_list = list(pad_sequences([tsst_list], maxlen=50, value=-1)[0])
        tbe_list = list(pad_sequences([tbe_list], maxlen=50, value=-1)[0])

        data_instance = np.array(tokenized_events + tbe_list + tsst_list).reshape(1, 50, 3)

        prediction = (model.predict(data_instance) > 0.5).astype(int)
        # NIEUW: Maak geheugen vrij na elke voorspelling
        K.clear_session()

        if prediction == 1:
            aggregated_data[session_id]['prediction'] = 1
            print('sending prediction')
            # NIEUW: Gebruik background task om niet te blokkeren
            asyncio.create_task(send_prediction({"sessionId": session_id}))

# Pydantic model voor de verwachte event data
class EventData(BaseModel):
    ga_stream_cookie: str
    event: str
    timestamp: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head>
            <title>API Instructions</title>
        </head>
        <body>
            <h1>Welcome to the Event API</h1>
            <p>This API accepts POST requests to the <code>/events</code> endpoint.</p>
            <h2>Instructions</h2>
            <p>To post an event, send a JSON payload to <code>/events</code> with the following structure:</p>
            <pre>{
    "ga_stream_cookie": "your_session_id_here",
    "event": "name_of_the_event",
    "timestamp": "YYYY-MM-DD HH:MM:SS"
    }</pre>
            <p>Example using cURL:</p>
            <pre>curl -X 'POST' \\
  'http://127.0.0.1:8000/events' \\
  -H 'accept: application/json' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "ga_stream_cookie": "12345",
    "event": "user_login",
    "timestamp": "2023-01-01 12:00:00"
    }'</pre>
            <p>Make sure your timestamp matches the expected format.</p>
        </body>
    </html>
    """

@app.post("/events")
async def receive_event(data: EventData):
    # Parse en valideer de binnenkomende data
    data = data.model_dump()
    task = asyncio.create_task(process_event(data))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

@app.get("/monitor_memory")
def monitor_memory():
    """Geeft een overzicht van de grootste geheugenverbruikers."""
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")
    response = []
    for stat in top_stats[:5]:
        response.append(f"{stat.size / 1024:.1f} KB - {stat.count} objecten - {stat.traceback.format()}")
    return {"top_memory_usage": response}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)