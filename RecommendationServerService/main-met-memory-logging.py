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
import requests
import httpx
from fastapi import HTTPException, status
import os

# Enable asyncio debug mode
import logging
import asyncio
import tracemalloc

asyncio.get_event_loop().set_debug(True)  # Enable asyncio debug mode
tracemalloc.start()  # Start monitoring memory allocations for debugging

# initialize server
app = FastAPI()
# initialize cache
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

# Function to log active asyncio tasks
async def log_active_tasks():
    while True:
        tasks = [t for t in asyncio.all_tasks() if not t.done()]
        logging.info(f"Active tasks: {len(tasks)}")
        for task in tasks:
            logging.debug(f"Task: {task.get_name()} - {task}")
        await asyncio.sleep(10)  # Log every 10 seconds

# Set a timeout for asyncio tasks
async def run_with_timeout(coro, timeout=30):
    try:
        return await asyncio.wait_for(coro, timeout)
    except asyncio.TimeoutError:
        logging.error("Task exceeded timeout and was cancelled.")
        raise HTTPException(status_code=500, detail="Task timeout")

# Function to send prediction to external API
async def send_prediction(prediction: dict):
    url = "https://api.rorix.nl/recommendation"
    headers = {
        "X-Apikey": "fPhVEkGpS7o3kQc41nwqdWE7VZYx6ZvY",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        response = await run_with_timeout(client.post(url, headers=headers, json=prediction))
        if response.status_code == 200:
            logging.info(response.json())
        elif response.status_code == 401:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed.")
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)

# Function to convert milliseconds to datetime
def milliseconds_to_datetime(ms):
    return datetime.fromtimestamp(ms / 1000.0, timezone.utc)

# Process event function with timeout and task monitoring
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
        tokenized_events = [tokenizer.texts_to_sequences([x])[0][0] for x in aggregated_data[session_id]['events']]
        tsst_list = [int(t.total_seconds()) for t in aggregated_data[session_id]['time_since_session_start']]
        tbe_list = [int(t.total_seconds()) for t in aggregated_data[session_id]['time_between_events']]

        tokenized_events = list(list(pad_sequences([tokenized_events], maxlen=50, value=-1))[0])
        tsst_list = list(list(pad_sequences([tsst_list], maxlen=50, value=-1))[0])
        tbe_list = list(list(pad_sequences([tbe_list], maxlen=50, value=-1))[0])

        data_instance = np.array(tokenized_events + tbe_list + tsst_list).reshape(1, 50, 3)

        # Get prediction
        prediction = (model.predict(data_instance) > 0.5).astype(int)
        if prediction == 1:
            aggregated_data[session_id]['prediction'] = 1
            logging.info('sending prediction')
            await send_prediction({"sessionId": session_id})

# Pydantic model for event data
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
        </body>
    </html>
    """

@app.post("/events")
async def receive_event(data: EventData):
    data = data.model_dump()
    task = asyncio.create_task(run_with_timeout(process_event(data)))
    task.set_name(f"process_event_{data['ga_stream_cookie']}")
    return {"message": "Event processing started"}

if __name__ == "__main__":
    # Start a task to log active asyncio tasks
    asyncio.create_task(log_active_tasks())
    uvicorn.run(app, port=8000)
