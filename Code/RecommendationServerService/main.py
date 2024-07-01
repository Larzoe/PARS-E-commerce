from fastapi import FastAPI, HTTPException
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
import requests
import httpx
from fastapi import HTTPException, status
import os

# initialize server
app = FastAPI()
# initialize chache
aggregated_data = TTLCache(maxsize=2000, ttl=1200)
# import tokenizer and neural network from directory
with open('tokenizer-test.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = tf.keras.models.load_model('LSTM-test.keras')

# configure app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.sanitairwinkel.nl/"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-Apikey", "Content-Type"],
)


async def send_prediction(prediction: dict):
    url = "https://api.rorix.nl/recommendation"
    headers = {
        "X-Apikey": "fPhVEkGpS7o3kQc41nwqdWE7VZYx6ZvY",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=prediction)
        if response.status_code == 200:
            print(response.json())
        elif response.status_code == 401:
            # Handling unauthenticated error specifically
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed.")
        else:
            # Generic error handling for other HTTP status codes
            raise HTTPException(status_code=response.status_code, detail=response.text)


def milliseconds_to_datetime(ms):
    return datetime.fromtimestamp(ms / 1000.0, timezone.utc)


# function to process events
async def process_event(event):
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
        # Tokenize the event_list to numbers = integer tokenization
        tokenized_events = [tokenizer.texts_to_sequences([x])[0][0] for x in aggregated_data[session_id]['events']]
        tsst_list = [int(t.total_seconds()) for t in aggregated_data[session_id]['time_since_session_start']]
        tbe_list = [int(t.total_seconds()) for t in aggregated_data[session_id]['time_between_events']]

        # Pre-Pad the tokenized text with 0's
        tokenized_events = list(list(pad_sequences([tokenized_events], maxlen=50, value=-1))[0])
        tsst_list = list(list(pad_sequences([tsst_list], maxlen=50, value=-1))[0])
        tbe_list = list(list(pad_sequences([tbe_list], maxlen=50, value=-1))[0])

        data_instance = np.array(tokenized_events + tbe_list + tsst_list).reshape(1, 50, 3)
        
        # get prediction
        prediction = (model.predict(data_instance) > 0.5).astype(int)
        if prediction == 1:
            aggregated_data[session_id]['prediction'] = 1
            
            # send prediction to API
            print('sending prediction')
            send_prediction({"sessionId":session_id})



# Define a Pydantic model for the expected data structure (if known)
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
    # Data is automatically parsed and validated against the EventData model
    data = data.model_dump()
    asyncio.create_task(process_event(data))
    

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
