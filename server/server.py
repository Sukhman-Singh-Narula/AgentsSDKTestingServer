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
import logging
import json
from datetime import datetime

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

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = getLogger(__name__)

# Create specialized loggers for different components
audio_logger = logging.getLogger("audio_processing")
agent_logger = logging.getLogger("agent_conversation")
transcription_logger = logging.getLogger("transcription")
esp32_logger = logging.getLogger("esp32_bridge")

# Set log levels
audio_logger.setLevel(logging.INFO)
agent_logger.setLevel(logging.INFO)
transcription_logger.setLevel(logging.INFO)
esp32_logger.setLevel(logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConversationTracker:
    """Track conversation state and progress through Spanish learning episodes"""
    
    def __init__(self):
        self.session_start = datetime.now()
        self.current_episode = 1
        self.current_act = 1
        self.spanish_words_learned = []
        self.spanish_words_attempted = []
        self.characters_met = []
        self.crystals_found = []
        self.conversation_turns = 0
        self.audio_exchanges = 0
        
    def log_episode_progress(self, agent_name: str, response_text: str):
        """Analyze agent response to track episode progress"""
        self.conversation_turns += 1
        
        # Extract episode number from agent name
        if "Episode 1" in agent_name:
            self.current_episode = 1
        elif "Episode 2" in agent_name:
            self.current_episode = 2
        elif "Episode 3" in agent_name:
            self.current_episode = 3
            
        # Detect Spanish words being taught/practiced
        spanish_indicators = {
            "hola": "Hello",
            "azul": "Blue", 
            "bien": "Good/Fine",
            "adiós": "Goodbye",
            "me llamo": "My name is",
            "como te llamas": "What is your name",
            "rojo": "Red",
            "amarillo": "Yellow",
            "arriba": "Up",
            "abajo": "Down"
        }
        
        response_lower = response_text.lower()
        for spanish, english in spanish_indicators.items():
            if spanish in response_lower and spanish not in self.spanish_words_learned:
                self.spanish_words_learned.append(f"{spanish} ({english})")
                agent_logger.info(f"🎓 NEW SPANISH WORD INTRODUCED: {spanish} = {english}")
        
        # Detect character introductions
        characters = ["Splash", "Sandy", "Luna", "Pip", "Rocky", "Pebble"]
        for character in characters:
            if character.lower() in response_lower and character not in self.characters_met:
                self.characters_met.append(character)
                agent_logger.info(f"👋 NEW CHARACTER MET: {character}")
        
        # Detect crystal discoveries
        crystals = ["Ocean Crystal", "Forest Crystal", "Sky Crystal"]
        for crystal in crystals:
            if crystal.lower() in response_lower and crystal not in self.crystals_found:
                self.crystals_found.append(crystal)
                agent_logger.info(f"💎 CRYSTAL FOUND: {crystal}")
        
        # Detect act progression keywords
        act_keywords = {
            1: ["ice breaker", "welcome", "what's your name"],
            2: ["world setup", "season introduction", "magical island"],
            3: ["character introduction", "meet", "beach arrival"],
            4: ["exploration", "bonding", "beach exploration"],
            5: ["crystal hunt", "victory", "discovery"]
        }
        
        # Log current session status
        elapsed_time = (datetime.now() - self.session_start).total_seconds()
        agent_logger.info(f"📊 SESSION STATUS - Episode: {self.current_episode}, "
                         f"Turns: {self.conversation_turns}, Elapsed: {elapsed_time:.1f}s")
        agent_logger.info(f"📚 Spanish Progress: {len(self.spanish_words_learned)} words learned")
        agent_logger.info(f"👥 Characters Met: {', '.join(self.characters_met) if self.characters_met else 'None'}")
        agent_logger.info(f"💎 Crystals Found: {', '.join(self.crystals_found) if self.crystals_found else 'None'}")

# Global conversation tracker
conversation_tracker = ConversationTracker()

class EnhancedWorkflow(VoiceWorkflowBase):
    def __init__(self, connection: WebsocketHelper):
        self.connection = connection

    async def run(self, input_text: str) -> AsyncIterator[str]:
        # Log user input with transcription
        transcription_logger.info(f"🎤 USER TRANSCRIPTION: '{input_text}'")
        
        conversation_history, latest_agent = await self.connection.show_user_input(
            input_text
        )

        # Log agent context
        agent_logger.info(f"🤖 ACTIVE AGENT: {latest_agent.name}")
        agent_logger.info(f"📜 CONVERSATION HISTORY LENGTH: {len(conversation_history)} messages")
        
        # Analyze user input for Spanish words
        spanish_words = ["hola", "azul", "bien", "adiós", "me llamo", "como te llamas", "rojo", "amarillo", "arriba", "abajo"]
        detected_spanish = [word for word in spanish_words if word in input_text.lower()]
        if detected_spanish:
            conversation_tracker.spanish_words_attempted.extend(detected_spanish)
            transcription_logger.info(f"🇪🇸 SPANISH DETECTED IN USER INPUT: {', '.join(detected_spanish)}")

        output = Runner.run_streamed(
            latest_agent,
            conversation_history,
        )

        agent_response = ""
        chunk_count = 0
        
        agent_logger.info("🎯 STARTING AGENT RESPONSE GENERATION...")

        async for event in output.stream_events():
            await self.connection.handle_new_item(event)

            if is_text_output(event):
                chunk_count += 1
                chunk_text = event.data.delta
                agent_response += chunk_text
                yield chunk_text
                
                # Log every 10th chunk to show progress
                if chunk_count % 10 == 0:
                    agent_logger.debug(f"📝 Response chunk #{chunk_count}: '{chunk_text[:50]}...'")

        # Log complete agent response
        agent_logger.info(f"✅ AGENT RESPONSE COMPLETE ({chunk_count} chunks)")
        agent_logger.info(f"💬 FULL AGENT RESPONSE: '{agent_response[:200]}{'...' if len(agent_response) > 200 else ''}'")
        
        # Track conversation progress
        conversation_tracker.log_episode_progress(latest_agent.name, agent_response)
        
        await self.connection.text_output_complete(output, is_done=True)

# Original WebSocket endpoint for new clients
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    with trace("Voice Agent Chat"):
        await websocket.accept()
        logger.info("🔗 NEW WEBSOCKET CONNECTION (/ws endpoint)")
        
        connection = WebsocketHelper(websocket, [], starting_agent)
        audio_buffer = []

        workflow = EnhancedWorkflow(connection)
        message_count = 0
        
        while True:
            try:
                message = await websocket.receive_json()
                message_count += 1
                
                # Log message type and basic info
                message_type = message.get("type", "unknown")
                logger.debug(f"📨 Message #{message_count}: {message_type}")
                
            except WebSocketDisconnect:
                logger.info("❌ CLIENT DISCONNECTED (/ws)")
                return

            # Handle text based messages
            if is_sync_message(message):
                connection.history = message["inputs"]
                if message.get("reset_agent", False):
                    connection.latest_agent = starting_agent
                    logger.info("🔄 AGENT RESET TO STARTING AGENT")
                    
            elif is_new_text_message(message):
                user_input = process_inputs(message, connection)
                logger.info(f"💭 PROCESSING TEXT MESSAGE: '{user_input}'")
                
                response_chunks = 0
                async for new_output_tokens in workflow.run(user_input):
                    response_chunks += 1
                    await connection.stream_response(new_output_tokens, is_text=True)
                
                logger.info(f"📤 TEXT RESPONSE SENT ({response_chunks} chunks)")

            # Handle a new audio chunk
            elif is_new_audio_chunk(message):
                audio_buffer.append(extract_audio_chunk(message))
                if len(audio_buffer) % 20 == 0:  # Log every 20 chunks
                    total_audio_seconds = sum(len(chunk) for chunk in audio_buffer) / 24000  # Assuming 24kHz
                    audio_logger.debug(f"🔉 Audio buffer: {len(audio_buffer)} chunks (~{total_audio_seconds:.1f}s)")

            # Send full audio to the agent
            elif is_audio_complete(message):
                total_audio_seconds = sum(len(chunk) for chunk in audio_buffer) / 24000
                audio_logger.info(f"🎙️ AUDIO COMPLETE - Processing {len(audio_buffer)} chunks ({total_audio_seconds:.1f}s of audio)")
                
                start_time = time.perf_counter()
                first_byte_time = None

                def transform_data(data):
                    nonlocal start_time, first_byte_time
                    if first_byte_time is None:
                        first_byte_time = time.perf_counter()
                        audio_logger.info(f"⚡ Time to first audio byte: {first_byte_time - start_time:.2f}s")
                    return data

                audio_input = concat_audio_chunks(audio_buffer)
                conversation_tracker.audio_exchanges += 1
                
                audio_logger.info(f"🎵 STARTING VOICE PIPELINE (Exchange #{conversation_tracker.audio_exchanges})")
                
                output = await VoicePipeline(
                    workflow=workflow,
                    config=VoicePipelineConfig(
                        tts_settings=TTSModelSettings(
                            buffer_size=512, transform_data=transform_data
                        )
                    )
                ).run(audio_input)
                
                audio_chunks_sent = 0
                async for event in output.stream():
                    await connection.send_audio_chunk(event)
                    audio_chunks_sent += 1
                    
                audio_logger.info(f"🔊 AUDIO RESPONSE COMPLETE - Sent {audio_chunks_sent} audio chunks")
                audio_buffer = []  # reset the audio buffer

# Audio format conversion functions for ESP32 compatibility
def convert_audio_for_esp32(audio_data: np.ndarray, source_rate: int = 24000, target_rate: int = 22050) -> bytes:
    """Convert audio from voice pipeline to ESP32-compatible format"""
    try:
        audio_logger.debug(f"🔧 Converting audio: {len(audio_data)} samples, {source_rate}Hz -> {target_rate}Hz")
        
        # Ensure audio is numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        # Ensure audio is 1D (mono)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)  # Convert to mono
            audio_logger.debug("🎵 Converted stereo to mono")
        
        # Resample if necessary with anti-aliasing
        if source_rate != target_rate:
            from scipy.signal import resample_poly
            
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
                audio_logger.debug(f"🎛️ Resampled with anti-aliasing: {up_factor}/{down_factor} ratio")
        
        # Apply gentle high-pass filter to remove DC offset and low-frequency noise
        if len(audio_data) > 100:
            sos_hp = signal.butter(2, 80, btype='high', fs=target_rate, output='sos')
            audio_data = signal.sosfilt(sos_hp, audio_data)
            audio_logger.debug("🎚️ Applied high-pass filter (80Hz)")
        
        # Normalize audio with headroom to prevent clipping
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.7
            audio_logger.debug(f"📊 Normalized audio (max: {max_val:.3f} -> 0.7)")
        
        # Apply gentle compression to reduce dynamic range
        audio_data = np.tanh(audio_data * 1.2) * 0.8
        
        # Convert to 16-bit PCM with dithering
        dither = np.random.normal(0, 0.5, len(audio_data))
        audio_data = audio_data + dither / 32768
        
        # Clip to valid range and convert to int16
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        audio_logger.debug(f"✅ Audio conversion complete: {len(audio_int16)} samples, {len(audio_int16.tobytes())} bytes")
        return audio_int16.tobytes()
        
    except Exception as e:
        audio_logger.error(f"❌ Error converting audio for ESP32: {e}")
        # Return silence on error
        silence = np.zeros(1024, dtype=np.int16)
        return silence.tobytes()

def chunk_audio_for_esp32(audio_bytes: bytes, chunk_size: int = 1024) -> list:
    """Break large audio data into ESP32-compatible chunks"""
    chunks = []
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        # Pad last chunk if necessary
        if len(chunk) < chunk_size and i + chunk_size >= len(audio_bytes):
            padding = bytes(chunk_size - len(chunk))
            chunk = chunk + padding
        chunks.append(chunk)
    
    audio_logger.debug(f"📦 Created {len(chunks)} chunks of {chunk_size} bytes each")
    return chunks

def detect_audio_sample_rate(audio_data: np.ndarray) -> int:
    """Detect the sample rate of audio data from voice pipeline"""
    data_length = len(audio_data)
    
    # Estimate based on typical chunk sizes and durations
    if data_length > 20000:
        return 24000
    elif data_length > 15000:
        return 22050
    else:
        return 16000

def analyze_esp32_audio_quality(pcm_data: bytes) -> dict:
    """Analyze incoming ESP32 audio for quality metrics"""
    if len(pcm_data) < 100:
        return {"error": "Insufficient audio data"}
    
    # Convert to numpy for analysis
    audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0
    
    # Calculate basic metrics
    rms = np.sqrt(np.mean(audio_float**2))
    peak = np.max(np.abs(audio_float))
    snr_estimate = 20 * np.log10(rms + 1e-10)  # Rough SNR estimate
    
    # Detect silence
    silence_threshold = 0.01
    is_mostly_silence = rms < silence_threshold
    
    return {
        "duration_seconds": len(audio_int16) / 8000,  # Assuming 8kHz ESP32 audio
        "samples": len(audio_int16),
        "rms_level": float(rms),
        "peak_level": float(peak),
        "snr_estimate_db": float(snr_estimate),
        "is_mostly_silence": bool(is_mostly_silence),
        "quality": "good" if rms > 0.05 and peak < 0.95 else "poor"
    }

# ESP32 Bridge Endpoint
@app.websocket("/upload")
async def esp32_bridge_endpoint(websocket: WebSocket):
    """Enhanced ESP32 bridge with detailed logging"""
    await websocket.accept()
    esp32_logger.info("🤖 ESP32 CONNECTED to bridge endpoint")
    
    # Create connection to internal voice agent system
    internal_connection = WebsocketHelper(websocket, [], starting_agent)
    audio_buffer = []
    
    workflow = EnhancedWorkflow(internal_connection)
    session_start = time.time()
    audio_chunks_received = 0
    total_bytes_received = 0
    
    try:
        while True:
            # Receive raw bytes from ESP32
            data = await websocket.receive_bytes()
            data_size = len(data)
            
            # Handle control signals from ESP32
            if data == b"NODATA":
                esp32_logger.warning("⚠️ ESP32: No useful audio data received")
                audio_buffer = []
                continue
                
            elif data == b"END":
                session_duration = time.time() - session_start
                esp32_logger.info(f"🎯 ESP32: Audio recording complete! Session: {session_duration:.1f}s, "
                                f"Chunks: {audio_chunks_received}, Bytes: {total_bytes_received}")
                
                # Process accumulated PCM data
                if audio_buffer:
                    try:
                        # Combine all PCM chunks
                        pcm_bytes = b''.join(audio_buffer)
                        esp32_logger.info(f"🎵 Processing {len(pcm_bytes)} bytes of ESP32 audio")
                        
                        # Analyze audio quality
                        quality_metrics = analyze_esp32_audio_quality(pcm_bytes)
                        esp32_logger.info(f"📊 Audio Quality: {quality_metrics['quality']} "
                                        f"(RMS: {quality_metrics['rms_level']:.3f}, "
                                        f"Peak: {quality_metrics['peak_level']:.3f}, "
                                        f"Duration: {quality_metrics['duration_seconds']:.1f}s)")
                        
                        if quality_metrics['is_mostly_silence']:
                            esp32_logger.warning("🔇 Warning: Audio appears to be mostly silence")
                        
                        # Convert PCM bytes to voice pipeline format
                        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                        audio_float32 = audio_int16.astype(np.float32) / 32768.0
                        audio_input = AudioInput(audio_float32)
                        
                        esp32_logger.info("🎙️ Starting voice pipeline processing...")
                        
                        start_time = time.perf_counter()
                        first_audio_time = None
                        
                        def transform_data(data):
                            nonlocal start_time, first_audio_time
                            if first_audio_time is None:
                                first_audio_time = time.perf_counter()
                                esp32_logger.info(f"⚡ Time to first TTS byte: {first_audio_time - start_time:.2f}s")
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
                        
                        esp32_logger.info("🔊 Streaming processed audio back to ESP32...")
                        
                        # Stream audio response back to ESP32
                        audio_chunks_sent = 0
                        total_audio_bytes = 0
                        accumulated_audio = bytearray()
                        
                        async for event in output.stream():
                            if isinstance(event, VoiceStreamEventAudio):
                                # Detect and convert audio
                                source_sample_rate = detect_audio_sample_rate(event.data)
                                esp32_audio_bytes = convert_audio_for_esp32(
                                    event.data, 
                                    source_rate=source_sample_rate, 
                                    target_rate=22050
                                )
                                accumulated_audio.extend(esp32_audio_bytes)
                        
                        # Send accumulated audio in chunks
                        if accumulated_audio:
                            esp32_logger.info(f"📦 Sending {len(accumulated_audio)} bytes as chunked audio to ESP32")
                            
                            audio_chunks = chunk_audio_for_esp32(bytes(accumulated_audio), chunk_size=1024)
                            esp32_logger.info(f"📤 Created {len(audio_chunks)} chunks for ESP32")
                            
                            # Send chunks with timing control
                            for i, chunk in enumerate(audio_chunks):
                                await websocket.send_bytes(chunk)
                                audio_chunks_sent += 1
                                total_audio_bytes += len(chunk)
                                
                                # Progress logging and flow control
                                if i % 20 == 0 and i > 0:
                                    await asyncio.sleep(0.01)  # Small delay
                                    esp32_logger.debug(f"📈 Progress: {audio_chunks_sent}/{len(audio_chunks)} chunks sent")
                        
                        esp32_logger.info(f"✅ Audio streaming complete! Sent {audio_chunks_sent} chunks "
                                        f"({total_audio_bytes} bytes) to ESP32")
                        
                        # Send EOF signal
                        await websocket.send_bytes(b"EOF")
                        esp32_logger.info("🏁 Sent EOF signal to ESP32")
                        
                    except Exception as e:
                        esp32_logger.error(f"❌ Error processing ESP32 audio: {e}")
                        await websocket.send_bytes(b"EOF")
                
                else:
                    esp32_logger.warning("⚠️ No audio data to process")
                    await websocket.send_bytes(b"EOF")
                
                # Reset for next recording
                audio_buffer = []
                audio_chunks_received = 0
                total_bytes_received = 0
                session_start = time.time()
                
            else:
                # Accumulate raw PCM audio data
                audio_buffer.append(data)
                audio_chunks_received += 1
                total_bytes_received += data_size
                
                # Periodic logging
                if audio_chunks_received % 100 == 0:
                    duration = time.time() - session_start
                    esp32_logger.debug(f"🎤 Recording: {audio_chunks_received} chunks, "
                                     f"{total_bytes_received} bytes, {duration:.1f}s")
                
    except WebSocketDisconnect:
        esp32_logger.info("❌ ESP32 DISCONNECTED")
    except Exception as e:
        esp32_logger.error(f"💥 ESP32 Bridge Error: {e}")
        try:
            await websocket.send_bytes(b"EOF")
        except:
            pass
    finally:
        session_duration = time.time() - session_start
        esp32_logger.info(f"🔚 ESP32 session ended. Duration: {session_duration:.1f}s, "
                         f"Final stats: {audio_chunks_received} chunks, {total_bytes_received} bytes")

# Health check endpoint with logging
@app.get("/health")
async def health_check():
    logger.info("🏥 Health check requested")
    return {
        "status": "healthy", 
        "endpoints": ["/ws", "/upload"],
        "conversation_stats": {
            "session_duration": str(datetime.now() - conversation_tracker.session_start),
            "conversation_turns": conversation_tracker.conversation_turns,
            "audio_exchanges": conversation_tracker.audio_exchanges,
            "spanish_words_learned": len(conversation_tracker.spanish_words_learned),
            "characters_met": len(conversation_tracker.characters_met),
            "crystals_found": len(conversation_tracker.crystals_found)
        }
    }

# Enhanced info endpoint
@app.get("/")
async def info():
    logger.info("ℹ️ Info endpoint accessed")
    return {
        "service": "Enhanced Voice Agent Server with ESP32 Bridge",
        "version": "2.0.0",
        "features": [
            "Detailed conversation logging",
            "Spanish learning progress tracking", 
            "Audio quality analysis",
            "ESP32 compatibility",
            "Real-time transcription logging"
        ],
        "endpoints": {
            "/ws": "WebSocket endpoint for new clients (JSON protocol)",
            "/upload": "WebSocket endpoint for ESP32 compatibility (raw bytes protocol)",
            "/health": "Health check with conversation stats",
            "/logs": "Access detailed logs"
        },
        "logging": {
            "transcription": "User speech transcription",
            "agent_conversation": "Agent responses and episode progress", 
            "audio_processing": "Audio conversion and quality metrics",
            "esp32_bridge": "ESP32 communication details"
        }
    }

# New endpoint to view recent logs
@app.get("/logs")
async def get_recent_logs():
    """Return recent conversation stats and progress"""
    return {
        "session_info": {
            "started": conversation_tracker.session_start.isoformat(),
            "duration": str(datetime.now() - conversation_tracker.session_start),
            "current_episode": conversation_tracker.current_episode
        },
        "progress": {
            "conversation_turns": conversation_tracker.conversation_turns,
            "audio_exchanges": conversation_tracker.audio_exchanges,
            "spanish_words_learned": conversation_tracker.spanish_words_learned,
            "spanish_words_attempted": conversation_tracker.spanish_words_attempted,
            "characters_met": conversation_tracker.characters_met,
            "crystals_found": conversation_tracker.crystals_found
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Banner with logging info
    print("🚀 Starting Enhanced Voice Agent Server with Detailed Logging")
    print("📡 ESP32 endpoint: ws://localhost:5000/upload")
    print("🌐 New clients endpoint: ws://localhost:5000/ws")
    print("📊 Health check: http://localhost:5000/health")
    print("📋 Logs endpoint: http://localhost:5000/logs")
    print("\n🔍 Available Log Categories:")
    print("  • transcription: User speech transcription")
    print("  • agent_conversation: Agent responses and progress")
    print("  • audio_processing: Audio quality and conversion")
    print("  • esp32_bridge: ESP32 communication details")
    print("\n🎯 Tracking Features:")
    print("  • Spanish learning progress")
    print("  • Episode and character progression")
    print("  • Audio quality metrics")
    print("  • Conversation flow analysis")
    
    uvicorn.run("server:app", host="0.0.0.0", port=5000, reload=True)
