import asyncio
import base64
import json
import os
import time
import queue
import threading
from asyncio import run_coroutine_threadsafe
from datetime import datetime
import random

import numpy as np
import sounddevice as sd
import streamlit as st
import tzlocal
import websockets
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from uuid import uuid4

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


HIDE_STREAMLIT_RUNNING_MAN_SCRIPT = """
<style>
    div[data-testid="stStatusWidget"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
</style>
"""

DEFAULT_INSTRUCTIONS = """You are a friendly Tutor who is welcoming a student. You are trying to learn as much about them as possible in around 10 minutes.

You will become their personal AI tutor, and you should let them know that.

------

Functions
You have a function available to you. YOU SHOULD BE CALLING THIS IF YOU NEED TO GET THE HISTORY OF THE USER CONVERSATION

These functions happen at the BEGINNING of each new response, so call them BEFORE you send the reply to best output and make student engaged within the conversation to talk with their interests and passions.

Function 1: getPreviousConversation
This is simple, when student asks some question or if they asks for somethinga about previous conversation then if you need some history of the previous conversation,
you should call this function to get the history and make this refine or rephrase as per student's current questions.

------

You will have a free flowing conversation with them making sure that you understand what makes them tick (
never say this verbatim),
and understand their biggest hobbies and passions, inside and outside of school. When they introduce something
new, you should dig deeper into it, understanding the "why" behind them.

Here is the flow:
An introduction, i.e. "Hello! My name is Sage. I'm pleased to meet you.
Then tell them a bit of a lo down on whats going to happen, that you're going talk with them about their 
interests, and that this is just a quick conversation that should take about 10 minutes. 

Loop over this as you ask questions and followups:
Use the function to display the question, and progress. 
"Ask Your Question Verbally"
(student must respond, once they do, ask next question and call question and progress function)

A final "Thank you!" and a good luck in your studies with me. See you next time!  

You should always ask small followups between questions to enhance the experience. 

You should use "I" and be very human, fun, and uplifting, but chill at the same time. You're just trying to be 
the student's friend/tutor and
get to know them. Be welcoming!

FINAL NOTES
YOU are driving the conversation. Never trail off with something that could lead to a pause, always be driving 
to the end of the session no matter what.

ALWAYS ALWAYS ALWAYS call both functions before you ask one of the 8 questions, but not for follow up questions. 

You should be extremely sensitive to their tone of voice, and should understand and match their emotional 
energy and slang whenever possible."""

SUMMARIZATION_PROMPT = """I'm sharing the conversation messages from a tutor, but I don't have any of the student’s messages. Please provide a concise summary by predicting the questions or statements based on the tutor’s responses.

I want to get a summary to understand the student's interests and passions. Don't forget to mention the topics, subjects, modules which were discussed in the conversation, so that I can get the personalized content.

NOTE: Don't add any extra content into the response.

[TUTOR CONVERSATIONS DATA]
```
<TUTOR_CONVERSATION_TRANSCRIPTED_DATA>
```
"""

OAI_LOGO_URL = "https://raw.githubusercontent.com/openai/openai-realtime-console/refs/heads/main/public/openai-logomark.svg"

OPENAI_WEBSOCKET_URL = "wss://api.openai.com/v1/realtime"

INDEX_NAME = "ai-poc"

pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def create_pinecone_index():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


def upsert_vector_embedding(data, embeddings):
    # Wait for the index to be ready
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

    index = pc.Index(INDEX_NAME)

    embeds = [record.embedding for record in embeddings]

    upsert_vectors = [{"id": f"vc-{uuid4()}", "values": embeds[0], "metadata": {"text": data}}]

    # # upsert to Pinecone
    index.upsert(vectors=upsert_vectors)


def create_openai_embedding(text: str):
    completion_response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": SUMMARIZATION_PROMPT.replace(
                    "<TUTOR_CONVERSATION_TRANSCRIPTED_DATA>", text
                ),
            },
        ],
    )

    response = openai_client.embeddings.create(
        input=json.dumps(
            {"full_conversation": text, "summary": completion_response.choices[0].message.content}
        ),
        model="text-embedding-ada-002",
    )

    upsert_vector_embedding(text, response.data)


def get_previous_conversation(user_input):
    json_text = json.loads(user_input).get("user_input")

    # create the query embedding
    xq = (
        openai_client.embeddings.create(input=json_text, model="text-embedding-ada-002")
        .data[0]
        .embedding
    )

    # Wait for the index to be ready
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

    index = pc.Index(INDEX_NAME)

    # query, returning the top 5 most similar results
    res = index.query(vector=[xq], top_k=5, include_metadata=True)

    return res


fns = {"getPreviousConversation": get_previous_conversation}


class SimpleRealtime:
    def __init__(self, event_loop=None, audio_buffer_cb=None, debug=False):
        self.ws = None
        self.logs = []
        self.debug = debug
        self.transcript = ""
        self.event_loop = event_loop
        self.url = OPENAI_WEBSOCKET_URL
        self._message_handler_task = None
        self.audio_buffer_cb = audio_buffer_cb

    def is_connected(self):
        return self.ws is not None and self.ws.paused is False

    def log_event(self, event_type, event):
        if self.debug:
            local_timezone = tzlocal.get_localzone()
            now = datetime.now(local_timezone).strftime("%H:%M:%S")
            msg = json.dumps(event)
            self.logs.append((now, event_type, msg))

        return True

    async def connect(self, model="gpt-4o-realtime-preview-2024-10-01"):
        if self.is_connected():
            raise Exception("Already connected")

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }

        self.ws = await websockets.connect(f"{self.url}?model={model}", additional_headers=headers)

        # Start the message handler in the same loop as the websocket
        self._message_handler_task = self.event_loop.create_task(self._message_handler())

        event = {
            "type": "session.update",
            "session": {
                "instructions": DEFAULT_INSTRUCTIONS,
                "modalities": ["text", "audio"],
                "tools": [
                    {
                        "type": "function",
                        "name": "getPreviousConversation",
                        "description": "If a student asks about his previous conversation, provide a concise summary.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "user_input": {
                                    "type": "string",
                                    "description": "The user's input",
                                },
                            },
                        },
                    },
                ],
            },
        }

        self.event_loop.create_task(self.ws.send(json.dumps(event)))
        self.event_loop.create_task(self.ws.send(json.dumps({"type": "response.create"})))

        return True

    async def _message_handler(self):
        try:
            while True:
                if not self.ws:
                    await asyncio.sleep(0.05)
                    continue

                try:
                    message = await asyncio.wait_for(self.ws.recv(), timeout=0.05)
                    data = json.loads(message)
                    self.receive(data)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break
        except Exception as e:
            print(f"Message handler error: {e}, line: {e.__traceback__.tb_lineno}")
            await self.disconnect()

    async def disconnect(self):
        create_openai_embedding(self.transcript)
        if self.ws:
            await self.ws.close()
            self.ws = None
        if self._message_handler_task:
            self._message_handler_task.cancel()
            try:
                await self._message_handler_task
            except asyncio.CancelledError:
                pass
        self._message_handler_task = None
        return True

    def handle_audio(self, event):
        if event.get("type") == "response.audio_transcript.delta":
            self.transcript += event.get("delta")

        if event.get("type") == "response.audio.delta" and self.audio_buffer_cb:
            b64_audio_chunk = event.get("delta")
            decoded_audio_chunk = base64.b64decode(b64_audio_chunk)
            pcm_audio_chunk = np.frombuffer(decoded_audio_chunk, dtype=np.int16)
            self.audio_buffer_cb(pcm_audio_chunk)

    def receive(self, event):
        self.log_event("server", event)
        # print("\n#######", event)
        if event.get("type") == "response.function_call_arguments.done":
            fun = fns.get(event.get("name"))
            result = fun(
                event.get("arguments")
                or {
                    "user_input": "Could you please provide a summary about our previous conversation?"
                }
            )
            event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": event.get("call_id"),
                    "output": result["matches"][0].get("metadata").get("text"),
                },
            }
            self.event_loop.create_task(self.ws.send(json.dumps(event)))
            # Have assistant respond after getting the results
            self.event_loop.create_task(self.ws.send(json.dumps({"type": "response.create"})))

        if "response.audio" in event.get("type"):
            self.handle_audio(event)
        return True

    def send(self, event_name, data=None):
        if not self.is_connected():
            raise Exception("RealtimeAPI is not connected")

        data = data or {}
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary")

        event = {"type": event_name, **data}

        self.log_event("client", event)

        self.event_loop.create_task(self.ws.send(json.dumps(event)))

        return True


def is_voice_present(indata, threshold=50):
    """Check if the RMS value of indata exceeds the threshold (indicating voice or sound)."""
    rms_value = np.sqrt(np.mean(indata**2))  # Compute RMS of the chunk
    print("***************", indata, rms_value)
    return rms_value > threshold


class StreamingAudioRecorder:
    def __init__(self, sample_rate=24_000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_thread = None

    def callback(self, indata, frames, time, status):
        """
        This will be called for each audio block
        that gets recorded.
        """
        global audio_buffer
        global voice_detected_counter

        # Check if the audio chunk contains voice or sound data
        if is_voice_present(indata):
            print("Voice data detected.")
            if voice_detected_counter >= 5:
                audio_buffer = np.array([], dtype=np.int16)
                voice_detected_counter = 0
            else:
                voice_detected_counter += 1
        else:
            print("Silence detected.")

        self.audio_queue.put(indata.copy())

    def start_recording(self):
        self.is_recording = True
        self.audio_thread = sd.InputStream(
            dtype="int16",
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.callback,
            blocksize=2_000,
        )
        self.audio_thread.start()

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.audio_thread.stop()
            self.audio_thread.close()

    def get_audio_chunk(self):
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None


# Streamlit page configuration
st.set_page_config(layout="wide")

# Global variables
audio_buffer = np.array([], dtype=np.int16)
buffer_lock = threading.Lock()
voice_detected_counter = 0


# Initialize session state variables if not already present
def init_session_state():
    if "audio_stream_started" not in st.session_state:
        st.session_state.audio_stream_started = False
    if "recorder" not in st.session_state:
        st.session_state.recorder = StreamingAudioRecorder()
    if "recording" not in st.session_state:
        st.session_state.recording = False


init_session_state()


# Audio buffer callback for real-time client
def audio_buffer_cb(pcm_audio_chunk):
    global audio_buffer
    with buffer_lock:
        audio_buffer = np.concatenate([audio_buffer, pcm_audio_chunk])


# SoundDevice playback callback
def sd_audio_cb(outdata, frames, time, status):
    global audio_buffer
    channels = 1
    with buffer_lock:
        if len(audio_buffer) >= frames:
            outdata[:] = audio_buffer[:frames].reshape(-1, channels)
            audio_buffer = audio_buffer[frames:]
        else:
            outdata.fill(0)


# Start audio stream for playback
def start_audio_stream():
    with sd.OutputStream(
        callback=sd_audio_cb,
        dtype="int16",
        samplerate=24000,
        channels=1,
        blocksize=2000,
    ):
        sd.sleep(int(10e6))


# Create event loop and start in a separate thread
@st.cache_resource(show_spinner=False)
def create_event_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    return loop, thread


st.session_state.event_loop, worker_thread = create_event_loop()


# Helper to run async functions in the global event loop
def run_async(coroutine):
    return run_coroutine_threadsafe(coroutine, st.session_state.event_loop).result()


# Setup and cache SimpleRealtime client
@st.cache_resource(show_spinner=False)
def setup_client():
    if "client" in st.session_state:
        return st.session_state.client
    return SimpleRealtime(
        event_loop=st.session_state.event_loop,
        audio_buffer_cb=audio_buffer_cb,
        debug=True,
    )


st.session_state.client = setup_client()


# Toggle recording state and handle actions
def toggle_recording():
    st.session_state.recording = not st.session_state.recording

    if st.session_state.recording:
        with st.spinner("Connecting..."):
            try:
                run_async(st.session_state.client.connect())
                if st.session_state.client.is_connected():
                    st.success("Connected to OpenAI Realtime API")
                else:
                    st.error("Failed to connect to OpenAI Realtime API")
            except Exception as e:
                st.error(f"Error connecting to OpenAI Realtime API: {str(e)}")
        st.session_state.recorder.start_recording()
    else:
        with st.spinner("Disconnecting..."):
            st.session_state.recorder.stop_recording()
            st.session_state.client.send("input_audio_buffer.commit")
            st.session_state.client.send("response.create")
            time.sleep(3)
            try:
                run_async(st.session_state.client.disconnect())
                st.success("Disconnected to OpenAI Realtime API")
            except Exception as e:
                st.error(f"Error disconnecting to OpenAI Realtime API: {str(e)}")


# Response area fragment
@st.fragment(run_every=1)
def response_area():
    st.markdown("**Conversation**")
    st.write(st.session_state.client.transcript)


# Audio player fragment
@st.fragment(run_every=1)
def audio_player():
    if not st.session_state.audio_stream_started:
        st.session_state.audio_stream_started = True
        start_audio_stream()


# Audio recorder fragment
@st.fragment(run_every=1)
def audio_recorder():
    if st.session_state.recording:
        while not st.session_state.recorder.audio_queue.empty():
            chunk = st.session_state.recorder.audio_queue.get()

            st.session_state.client.send(
                "input_audio_buffer.append", {"audio": base64.b64encode(chunk).decode()}
            )


# Clear input text area callback
def clear_input_cb():
    st.session_state.last_input = st.session_state.input_text_area
    st.session_state.input_text_area = ""


# Main Streamlit app function
def st_app():
    st.markdown(HIDE_STREAMLIT_RUNNING_MAN_SCRIPT, unsafe_allow_html=True)
    st.markdown(
        f"<img src='{OAI_LOGO_URL}' width='30px'/> **Realtime Console**",
        unsafe_allow_html=True,
    )

    with st.container(height=300, key="response_container"):
        response_area()

    button_text = "Stop Recording" if st.session_state.recording else "Start Coversation"
    st.button(button_text, on_click=toggle_recording, type="primary")

    audio_player()
    audio_recorder()


if __name__ == "__main__":
    # create_pinecone_index()  # Run only once for creating index
    st_app()
