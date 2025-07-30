import time
from collections.abc import AsyncIterator
from logging import getLogger
from typing import Any, Dict
import numpy as np
from scipy import signal
from scipy.signal import resample_poly
import io
import wave
import asyncio

from agents import Runner, trace
from agents.voice import (
    TTSModelSettings,
    VoicePipeline,
    VoicePipelineConfig,
    VoiceWorkflowBase,
    AudioInput,
    VoiceStreamEventAudio
)
from app.agent_config import starting_agent
from app.utils import (
    WebsocketHelper,
    concat_audio_chunks,
    extract_audio_chunk,
    is_audio_complete,
    is_new_audio_chunk,
    is_new_text_message,
    is_sync_message,
    is_text_output,
    process_inputs,
)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from dotenv import load_dotenv

# When .env file is present, it will override the environment variables
load_dotenv(dotenv_path="../.env", override=True)

app = FastAPI()

logger = getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Workflow(VoiceWorkflowBase):
    def __init__(self, connection: WebsocketHelper):
        self.connection = connection

    async def run(self, input_text: str) -> AsyncIterator[str]:
        conversation_history, latest_agent = await self.connection.show_user_input(
            input_text
        )

        output = Runner.run_streamed(
            latest_agent,
            conversation_history,
        )

        async for event in output.stream_events():
            await self.connection.handle_new_item(event)

            if is_text_output(event):
                yield event.data.delta  # type: ignore

        await self.connection.text_output_complete(output, is_done=True)

# Original WebSocket endpoint for new clients
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    with trace("Voice Agent Chat"):
        await websocket.accept()
        connection = WebsocketHelper(websocket, [], starting_agent)
        audio_buffer = []

        workflow = Workflow(connection)
        while True:
            try:
                message = await websocket.receive_json()
            except WebSocketDisconnect:
                print("Client disconnected")
                return

            # Handle text based messages
            if is_sync_message(message):
                connection.history = message["inputs"]
                if message.get("reset_agent", False):
                    connection.latest_agent = starting_agent
            elif is_new_text_message(message):
                user_input = process_inputs(message, connection)
                async for new_output_tokens in workflow.run(user_input):
                    await connection.stream_response(new_output_tokens, is_text=True)

            # Handle a new audio chunk
            elif is_new_audio_chunk(message):
                audio_buffer.append(extract_audio_chunk(message))

            # Send full audio to the agent
            elif is_audio_complete(message):
                start_time = time.perf_counter()

                def transform_data(data):
                    nonlocal start_time
                    if start_time:
                        print(
                            f"Time taken to first byte: {time.perf_counter() - start_time}s"
                        )
                        start_time = None
                    return data

                audio_input = concat_audio_chunks(audio_buffer)
                output = await VoicePipeline(
                    workflow=workflow,
                    config=VoicePipelineConfig(
                        tts_settings=TTSModelSettings(
                            buffer_size=512, transform_data=transform_data
                        )
                    ),
                ).run(audio_input)
                async for event in output.stream():
                    await connection.send_audio_chunk(event)

                audio_buffer = []  # reset the audio buffer

# Audio format conversion functions for ESP32 compatibility
def convert_audio_for_esp32(audio_data: np.ndarray, source_rate: int = 24000, target_rate: int = 22050) -> bytes:
    """
    Convert audio from voice pipeline to ESP32-compatible format
    ESP32 expects: 22050 Hz, Mono, 16-bit PCM
    """
    try:
        # Ensure audio is numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        # Ensure audio is 1D (mono)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)  # Convert to mono
        
        # Resample if necessary with anti-aliasing
        if source_rate != target_rate:
            # Use high-quality resampling with anti-aliasing filter
            from scipy.signal import resample_poly
            
            # Calculate rational resampling factors to avoid artifacts
            gcd_val = np.gcd(source_rate, target_rate)
            up_factor = target_rate // gcd_val
            down_factor = source_rate // gcd_val
            
            if up_factor != 1 or down_factor != 1:
                # Apply anti-aliasing filter before resampling
                nyquist = min(source_rate, target_rate) / 2
                cutoff = 0.8 * nyquist
                sos = signal.butter(6, cutoff, btype='low', fs=source_rate, output='sos')
                audio_data = signal.sosfilt(sos, audio_data)
                
                # High-quality rational resampling
                audio_data = resample_poly(audio_data, up_factor, down_factor)
                print(f"High-quality resampled audio from {source_rate}Hz to {target_rate}Hz")
            else:
                print(f"No resampling needed: {source_rate}Hz")
        
        # Apply gentle high-pass filter to remove DC offset and low-frequency noise
        if len(audio_data) > 100:  # Only if we have enough samples
            sos_hp = signal.butter(2, 80, btype='high', fs=target_rate, output='sos')
            audio_data = signal.sosfilt(sos_hp, audio_data)
        
        # Normalize audio with headroom to prevent clipping
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            # Normalize to 70% to prevent clipping and distortion
            audio_data = audio_data / max_val * 0.7
        
        # Apply gentle compression to reduce dynamic range
        audio_data = np.tanh(audio_data * 1.2) * 0.8
        
        # Convert to 16-bit PCM with dithering to reduce quantization noise
        # Add small amount of dither noise
        dither = np.random.normal(0, 0.5, len(audio_data))
        audio_data = audio_data + dither / 32768
        
        # Clip to valid range and convert to int16
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        return audio_int16.tobytes()
        
    except Exception as e:
        print(f"Error converting audio for ESP32: {e}")
        # Return silence on error
        silence = np.zeros(1024, dtype=np.int16)
        return silence.tobytes()

def chunk_audio_for_esp32(audio_bytes: bytes, chunk_size: int = 1024) -> list:
    """
    Break large audio data into ESP32-compatible chunks
    ESP32 I2S works best with smaller chunks to prevent buffer overflow
    """
    chunks = []
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        # Pad last chunk if necessary to maintain consistent timing
        if len(chunk) < chunk_size and i + chunk_size >= len(audio_bytes):
            # Pad with silence (zeros) instead of random data
            padding = bytes(chunk_size - len(chunk))
            chunk = chunk + padding
        chunks.append(chunk)
    return chunks

def detect_audio_sample_rate(audio_data: np.ndarray) -> int:
    """
    Try to detect the sample rate of audio data from voice pipeline
    Common rates: 16000, 22050, 24000, 44100, 48000
    """
    # This is a simple heuristic - in practice, you might need to get this from the voice pipeline
    data_length = len(audio_data)
    
    # Estimate based on typical chunk sizes and durations
    if data_length > 20000:  # Likely 24kHz or higher
        return 24000
    elif data_length > 15000:  # Likely 22kHz
        return 22050
    else:  # Likely 16kHz
        return 16000

# ESP32 Bridge Endpoint - Compatible with existing firmware
@app.websocket("/upload")
async def esp32_bridge_endpoint(websocket: WebSocket):
    """Bridge endpoint for ESP32 compatibility - no firmware changes needed"""
    await websocket.accept()
    print("ESP32 connected to bridge endpoint")
    
    # Create connection to internal voice agent system
    internal_connection = WebsocketHelper(websocket, [], starting_agent)
    audio_buffer = []
    
    workflow = Workflow(internal_connection)
    
    try:
        while True:
            # Receive raw bytes from ESP32 (existing firmware format)
            data = await websocket.receive_bytes()
            
            # Handle control signals from ESP32
            if data == b"NODATA":
                print("ESP32: No useful audio data received")
                audio_buffer = []  # Clear buffer
                break
                
            elif data == b"END":
                print("ESP32: Audio recording complete, processing...")
                
                # Process accumulated PCM data through voice pipeline
                if audio_buffer:
                    try:
                        # Combine all PCM chunks
                        pcm_bytes = b''.join(audio_buffer)
                        print(f"Processing {len(pcm_bytes)} bytes of audio data")
                        
                        # Convert PCM bytes to numpy array (format expected by voice pipeline)
                        # ESP32 sends 16-bit PCM at 8kHz, convert to float32 normalized [-1, 1]
                        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                        audio_float32 = audio_int16.astype(np.float32) / 32768.0
                        audio_input = AudioInput(audio_float32)
                        
                        print("Starting voice pipeline processing...")
                        
                        # Process through the voice pipeline
                        start_time = time.perf_counter()
                        
                        def transform_data(data):
                            nonlocal start_time
                            if start_time:
                                print(f"Time taken to first audio byte: {time.perf_counter() - start_time:.2f}s")
                                start_time = None
                            return data
                        
                        output = await VoicePipeline(
                            workflow=workflow,
                            config=VoicePipelineConfig(
                                tts_settings=TTSModelSettings(
                                    buffer_size=512,
                                    transform_data=transform_data
                                )
                            )
                        ).run(audio_input)
                        
                        print("Streaming audio response back to ESP32...")
                        
                        # Stream audio response back to ESP32 as raw bytes
                        audio_chunks_sent = 0
                        total_audio_bytes = 0
                        
                        # Buffer to accumulate audio data before chunking
                        accumulated_audio = bytearray()
                        
                        async for event in output.stream():
                            if isinstance(event, VoiceStreamEventAudio):
                                # Detect sample rate of incoming audio
                                source_sample_rate = detect_audio_sample_rate(event.data)
                                
                                # Convert audio to ESP32-compatible format
                                # ESP32 DAC expects: 22050 Hz, Mono, 16-bit PCM
                                esp32_audio_bytes = convert_audio_for_esp32(
                                    event.data, 
                                    source_rate=source_sample_rate, 
                                    target_rate=22050  # ESP32 DAC sample rate
                                )
                                
                                # Accumulate audio data
                                accumulated_audio.extend(esp32_audio_bytes)
                        
                        # Now send the accumulated audio in ESP32-compatible chunks
                        if accumulated_audio:
                            print(f"Breaking {len(accumulated_audio)} bytes into ESP32-compatible chunks...")
                            
                            # Break into 1024-byte chunks (matches ESP32 I2S buffer size)
                            audio_chunks = chunk_audio_for_esp32(bytes(accumulated_audio), chunk_size=1024)
                            
                            print(f"Created {len(audio_chunks)} chunks of 1024 bytes each")
                            
                            # Send chunks with small delay to prevent I2S buffer overflow
                            for i, chunk in enumerate(audio_chunks):
                                await websocket.send_bytes(chunk)
                                audio_chunks_sent += 1
                                total_audio_bytes += len(chunk)
                                
                                # Small delay every 10 chunks to prevent buffer overflow
                                if i % 10 == 0 and i > 0:
                                    await asyncio.sleep(0.01)  # 10ms delay
                                    print(f"Sent {audio_chunks_sent} chunks ({total_audio_bytes} bytes) to ESP32")
                        
                        print(f"Audio streaming complete: {audio_chunks_sent} chunks, {total_audio_bytes} bytes total")
                        
                        # Send EOF signal to ESP32 (expected by firmware)
                        await websocket.send_bytes(b"EOF")
                        print("Sent EOF signal to ESP32")
                        
                    except Exception as e:
                        print(f"Error processing audio: {e}")
                        # Send EOF even on error to prevent ESP32 from hanging
                        await websocket.send_bytes(b"EOF")
                
                else:
                    print("No audio data to process")
                    await websocket.send_bytes(b"EOF")
                
                # Reset buffer for next recording
                audio_buffer = []
                
            else:
                # Accumulate raw PCM audio data from ESP32
                audio_buffer.append(data)
                if len(audio_buffer) % 50 == 0:  # Log every 50 chunks
                    total_bytes = sum(len(chunk) for chunk in audio_buffer)
                    print(f"Accumulated {len(audio_buffer)} audio chunks ({total_bytes} bytes)")
                
    except WebSocketDisconnect:
        print("ESP32 disconnected")
    except Exception as e:
        print(f"ESP32 Bridge Error: {e}")
        try:
            await websocket.send_bytes(b"EOF")  # Try to send EOF on error
        except:
            pass
    finally:
        print("ESP32 bridge connection closed")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "endpoints": ["/ws", "/upload"]}

# Info endpoint
@app.get("/")
async def info():
    return {
        "service": "Voice Agent Server with ESP32 Bridge",
        "endpoints": {
            "/ws": "WebSocket endpoint for new clients (JSON protocol)",
            "/upload": "WebSocket endpoint for ESP32 compatibility (raw bytes protocol)",
            "/health": "Health check endpoint"
        },
        "esp32_compatibility": "Enabled - no firmware changes required"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Voice Agent Server with ESP32 Bridge")
    print("📡 ESP32 endpoint: ws://localhost:8000/upload")
    print("🌐 New clients endpoint: ws://localhost:8000/ws")
    uvicorn.run("server:app", host="0.0.0.0", port=5000, reload=True)
