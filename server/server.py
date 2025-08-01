import time
from collections.abc import AsyncIterator
from logging import getLogger
from typing import Any, Dict, Optional
import numpy as np
from scipy import signal
from scipy.signal import resample_poly
import io
import wave
import asyncio
import logging
import json
import uuid
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import SERVER_TIMESTAMP

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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from dotenv import load_dotenv

# When .env file is present, it will override the environment variables
load_dotenv(dotenv_path="../.env", override=True)

# Initialize Firebase
def init_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        # Check if Firebase is already initialized
        firebase_admin.get_app()
        print("🔥 Firebase already initialized")
    except ValueError:
        # Initialize Firebase if not already done
        cred_path = os.path.join(os.path.dirname(__file__), "firebase-credentials.json")
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            print("🔥 Firebase initialized successfully")
        else:
            print("❌ Firebase credentials file not found")
            raise FileNotFoundError("firebase-credentials.json not found")

# Initialize Firebase
try:
    init_firebase()
    db = firestore.client()
    print("🔥 Firestore client initialized")
except Exception as e:
    print(f"❌ Firebase initialization failed: {e}")
    db = None

# Pydantic models for API requests
class UserRegistration(BaseModel):
    device_id: str
    name: str
    age: int

class UserUpdate(BaseModel):
    name: str

# User Management System
@dataclass
class UserProgress:
    """Track user's learning progress"""
    episode: int = 1
    episodes_completed: int = 0
    season: int = 1

@dataclass 
class UserData:
    """Complete user profile and metrics"""
    user_id: str
    created_at: datetime
    device_id: str
    last_active: datetime
    last_completed_episode: Optional[datetime] = None
    name: str = "Unknown User"
    age: int = 0
    progress: UserProgress = field(default_factory=UserProgress)
    topics_learnt: list = field(default_factory=list)
    total_time: float = 0.0
    words_learnt: list = field(default_factory=list)
    status: str = "active"
    session_start_time: Optional[datetime] = None
    
    def update_last_active(self):
        """Update last active timestamp"""
        self.last_active = datetime.now(timezone.utc)
    
    def start_session(self):
        """Start a new session"""
        self.session_start_time = datetime.now(timezone.utc)
        self.update_last_active()
    
    def end_session(self):
        """End current session and update total time"""
        if self.session_start_time:
            session_duration = (datetime.now(timezone.utc) - self.session_start_time).total_seconds()
            self.total_time += session_duration
            self.session_start_time = None
        self.update_last_active()
    
    def complete_episode(self, episode_number: int):
        """Mark episode as completed"""
        if episode_number > self.progress.episodes_completed:
            self.progress.episodes_completed = episode_number
            self.last_completed_episode = datetime.now(timezone.utc)
            self.progress.episode = episode_number + 1
    
    def add_word_learned(self, word: str):
        """Add a new word to learned vocabulary"""
        if word.lower() not in [w.lower() for w in self.words_learnt]:
            self.words_learnt.append(word.lower())
    
    def add_topic_learned(self, topic: str):
        """Add a new topic to learned topics"""
        if topic.lower() not in [t.lower() for t in self.topics_learnt]:
            self.topics_learnt.append(topic.lower())
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "device_id": self.device_id,
            "last_active": self.last_active.isoformat(),
            "last_completed_episode": self.last_completed_episode.isoformat() if self.last_completed_episode else None,
            "name": self.name,
            "age": self.age,
            "progress": {
                "episode": self.progress.episode,
                "episodes_completed": self.progress.episodes_completed,
                "season": self.progress.season
            },
            "topics_learnt": self.topics_learnt,
            "total_time": self.total_time,
            "words_learnt": self.words_learnt,
            "status": self.status
        }

class UserManager:
    """Manage all users and their data"""
    
    def __init__(self, data_file: str = "user_data.json"):
        self.data_file = Path(data_file)
        self.users: Dict[str, UserData] = {}
        self.device_to_user: Dict[str, str] = {}  # Map device_id to user_id
        self.db = db  # Firebase Firestore client
        
        # Create specialized logger first
        self.logger = logging.getLogger("user_manager")
        self.logger.setLevel(logging.INFO)
        
        self.load_users()
    
    def load_users(self):
        """Load users from JSON file"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    
                for user_data in data.get('users', []):
                    user = UserData(
                        user_id=user_data['user_id'],
                        created_at=datetime.fromisoformat(user_data['created_at']),
                        device_id=user_data['device_id'],
                        last_active=datetime.fromisoformat(user_data['last_active']),
                        last_completed_episode=datetime.fromisoformat(user_data['last_completed_episode']) if user_data.get('last_completed_episode') else None,
                        name=user_data.get('name', 'Unknown User'),
                        age=user_data.get('age', 0),
                        progress=UserProgress(
                            episode=user_data['progress']['episode'],
                            episodes_completed=user_data['progress']['episodes_completed'],
                            season=user_data['progress']['season']
                        ),
                        topics_learnt=user_data.get('topics_learnt', []),
                        total_time=user_data.get('total_time', 0.0),
                        words_learnt=user_data.get('words_learnt', []),
                        status=user_data.get('status', 'active')
                    )
                    self.users[user.user_id] = user
                    self.device_to_user[user.device_id] = user.user_id
                    
                self.logger.info(f"📚 Loaded {len(self.users)} users from {self.data_file}")
            except Exception as e:
                self.logger.error(f"❌ Error loading users: {e}")
    
    def save_users(self):
        """Save users to JSON file"""
        try:
            data = {
                "users": [user.to_dict() for user in self.users.values()],
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug(f"💾 Saved {len(self.users)} users to {self.data_file}")
        except Exception as e:
            self.logger.error(f"❌ Error saving users: {e}")
    
    def save_user_to_firebase(self, user: UserData):
        """Save or update user data in Firebase"""
        if not self.db:
            self.logger.warning("🔥 Firebase not initialized, skipping save")
            return
        
        try:
            user_data = {
                "device_id": user.device_id,
                "name": user.name,
                "age": user.age,
                "created_at": user.created_at,
                "last_active": user.last_active,
                "last_completed_episode": user.last_completed_episode,
                "progress": {
                    "episode": user.progress.episode,
                    "episodes_completed": user.progress.episodes_completed,
                    "season": user.progress.season
                },
                "topics_learnt": user.topics_learnt,
                "total_time": user.total_time,
                "words_learnt": user.words_learnt,
                "status": user.status
            }
            
            # Use device_id as document ID in users collection
            doc_ref = self.db.collection('users').document(user.device_id)
            doc_ref.set(user_data, merge=True)
            
            self.logger.info(f"🔥 User {user.name} ({user.device_id}) saved to Firebase")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving user to Firebase: {e}")
    
    def load_user_from_firebase(self, device_id: str) -> Optional[UserData]:
        """Load user data from Firebase by device_id"""
        if not self.db:
            return None
        
        try:
            doc_ref = self.db.collection('users').document(device_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                user = UserData(
                    user_id=str(uuid.uuid4()),  # Generate new user_id for local use
                    device_id=data['device_id'],
                    name=data.get('name', 'Unknown User'),
                    age=data.get('age', 0),
                    created_at=data['created_at'] if isinstance(data['created_at'], datetime) else datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
                    last_active=data['last_active'] if isinstance(data['last_active'], datetime) else datetime.fromisoformat(data['last_active'].replace('Z', '+00:00')),
                    last_completed_episode=data.get('last_completed_episode') if data.get('last_completed_episode') else None,
                    progress=UserProgress(
                        episode=data['progress']['episode'],
                        episodes_completed=data['progress']['episodes_completed'],
                        season=data['progress']['season']
                    ),
                    topics_learnt=data.get('topics_learnt', []),
                    total_time=data.get('total_time', 0.0),
                    words_learnt=data.get('words_learnt', []),
                    status=data.get('status', 'active')
                )
                
                self.logger.info(f"🔥 User {user.name} ({device_id}) loaded from Firebase")
                return user
                
        except Exception as e:
            self.logger.error(f"❌ Error loading user from Firebase: {e}")
        
        return None
    
    def register_user(self, device_id: str, name: str, age: int) -> UserData:
        """Register a new user with device_id, name, and age"""
        # Check if user already exists locally
        if device_id in self.device_to_user:
            existing_user = self.users[self.device_to_user[device_id]]
            # Update existing user info
            existing_user.name = name
            existing_user.age = age
            existing_user.update_last_active()
            self.save_users()
            self.save_user_to_firebase(existing_user)
            self.logger.info(f"📝 Updated existing user: {name} ({device_id})")
            return existing_user
        
        # Check Firebase for existing user
        firebase_user = self.load_user_from_firebase(device_id)
        if firebase_user:
            # Update with new info
            firebase_user.name = name
            firebase_user.age = age
            firebase_user.update_last_active()
            
            # Add to local cache
            self.users[firebase_user.user_id] = firebase_user
            self.device_to_user[device_id] = firebase_user.user_id
            self.save_users()
            self.save_user_to_firebase(firebase_user)
            return firebase_user
        
        # Create completely new user
        user_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        user = UserData(
            user_id=user_id,
            created_at=now,
            device_id=device_id,
            last_active=now,
            name=name,
            age=age
        )
        
        self.users[user_id] = user
        self.device_to_user[device_id] = user_id
        self.save_users()
        self.save_user_to_firebase(user)
        
        self.logger.info(f"🆕 New user registered: {name} (age {age}) with device {device_id}")
        return user
    
    def get_or_create_user(self, device_id: str, name: str = None, age: int = 0) -> UserData:
        """Get existing user by device_id or create new one"""
        if device_id in self.device_to_user:
            user_id = self.device_to_user[device_id]
            user = self.users[user_id]
            user.update_last_active()
            # Update Firebase on each access
            self.save_user_to_firebase(user)
            self.logger.info(f"👤 Existing user connected: {user.name} ({user.user_id[:8]}...)")
            return user
        else:
            # Check Firebase first
            firebase_user = self.load_user_from_firebase(device_id)
            if firebase_user:
                self.users[firebase_user.user_id] = firebase_user
                self.device_to_user[device_id] = firebase_user.user_id
                self.save_users()
                return firebase_user
            
            # Create new user
            user_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            
            user = UserData(
                user_id=user_id,
                created_at=now,
                device_id=device_id,
                last_active=now,
                name=name or f"User_{device_id[:8]}",
                age=age
            )
            
            self.users[user_id] = user
            self.device_to_user[device_id] = user_id
            self.save_users()
            self.save_user_to_firebase(user)
            
            self.logger.info(f"🆕 New user created: {user.name} ({user_id[:8]}...) with device {device_id}")
            return user
    
    def update_user_conversation(self, user_id: str, agent_name: str, response_text: str):
        """Update user metrics based on conversation"""
        if user_id not in self.users:
            return
            
        user = self.users[user_id]
        user.update_last_active()
        
        # Extract episode progress
        if "Episode 1" in agent_name:
            current_ep = 1
        elif "Episode 2" in agent_name:
            current_ep = 2
        elif "Episode 3" in agent_name:
            current_ep = 3
        else:
            current_ep = user.progress.episode
        
        # Update episode progress
        if current_ep > user.progress.episode:
            user.progress.episode = current_ep
        
        # Detect Spanish words being taught
        spanish_words = {
            "hola": "hello", "azul": "blue", "bien": "good", "adiós": "goodbye",
            "me llamo": "my name is", "como te llamas": "what is your name",
            "rojo": "red", "amarillo": "yellow", "arriba": "up", "abajo": "down",
            "gato": "cat", "perro": "dog", "casa": "house", "agua": "water",
            "comida": "food", "familia": "family", "amigo": "friend"
        }
        
        response_lower = response_text.lower()
        for spanish, english in spanish_words.items():
            if spanish in response_lower:
                user.add_word_learned(spanish)
                user.add_word_learned(english)
        
        # Detect topics being covered
        topics = {
            "greetings": ["hola", "hello", "me llamo", "como te llamas", "nice to meet"],
            "colors": ["azul", "rojo", "amarillo", "blue", "red", "yellow", "color"],
            "animals": ["gato", "perro", "cat", "dog", "animal"],
            "family": ["familia", "family", "madre", "father", "hermano"],
            "food": ["comida", "food", "agua", "water", "eat", "drink"],
            "directions": ["arriba", "abajo", "up", "down", "left", "right"]
        }
        
        for topic, keywords in topics.items():
            if any(keyword in response_lower for keyword in keywords):
                user.add_topic_learned(topic)
        
        # Check for episode completion indicators
        completion_phrases = ["episode complete", "congratulations", "you did it", "finished"]
        if any(phrase in response_lower for phrase in completion_phrases):
            user.complete_episode(current_ep)
            self.logger.info(f"🎉 User {user.name} completed episode {current_ep}!")
        
        # Save updates periodically
        if len(user.words_learnt) % 5 == 0:  # Save every 5 new words
            self.save_users()
            self.save_user_to_firebase(user)  # Also save to Firebase
        
        self.logger.info(f"📊 User {user.name}: Episode {user.progress.episode}, "
                        f"{len(user.words_learnt)} words, {len(user.topics_learnt)} topics, "
                        f"{user.total_time:.1f}s total time")
    
    def get_user_stats(self) -> dict:
        """Get overall user statistics"""
        return {
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.status == "active"]),
            "total_words_learned": sum(len(u.words_learnt) for u in self.users.values()),
            "total_time": sum(u.total_time for u in self.users.values()),
            "episodes_completed": sum(u.progress.episodes_completed for u in self.users.values())
        }

# Global user manager
user_manager = UserManager()

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
    def __init__(self, connection: WebsocketHelper, user: UserData = None):
        self.connection = connection
        self.user = user

    async def run(self, input_text: str) -> AsyncIterator[str]:
        # Log user input with transcription
        transcription_logger.info(f"🎤 USER TRANSCRIPTION: '{input_text}'")
        
        # If we have a user, start their session
        if self.user:
            if not self.user.session_start_time:
                self.user.start_session()
        
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
        
        # Update user metrics if we have a user
        if self.user:
            user_manager.update_user_conversation(self.user.user_id, latest_agent.name, agent_response)
        
        await self.connection.text_output_complete(output, is_done=True)

# Original WebSocket endpoint for new clients
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    with trace("Voice Agent Chat"):
        await websocket.accept()
        logger.info("🔗 NEW WEBSOCKET CONNECTION (/ws endpoint)")
        
        # Wait for initial connection message with device_id and name
        user = None
        try:
            init_message = await websocket.receive_json()
            device_id = init_message.get("device_id", f"web_{int(time.time())}")
            user_name = init_message.get("name", None)
            
            # Get or create user
            user = user_manager.get_or_create_user(device_id, user_name)
            logger.info(f"👤 User connected: {user.name} (ID: {user.user_id[:8]}...)")
            
            # Send user info back to client
            await websocket.send_json({
                "type": "user.connected",
                "user_id": user.user_id,
                "name": user.name,
                "progress": {
                    "episode": user.progress.episode,
                    "episodes_completed": user.progress.episodes_completed,
                    "season": user.progress.season
                },
                "stats": {
                    "words_learnt": len(user.words_learnt),
                    "topics_learnt": len(user.topics_learnt),
                    "total_time": user.total_time
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Error during user initialization: {e}")
            # Create anonymous user as fallback
            device_id = f"anonymous_{int(time.time())}"
            user = user_manager.get_or_create_user(device_id)
        
        connection = WebsocketHelper(websocket, [], starting_agent)
        audio_buffer = []

        workflow = EnhancedWorkflow(connection, user)
        message_count = 0
        
        try:
            while True:
                try:
                    message = await websocket.receive_json()
                    message_count += 1
                    
                    # Log message type and basic info
                    message_type = message.get("type", "unknown")
                    logger.debug(f"📨 Message #{message_count}: {message_type}")
                    
                except WebSocketDisconnect:
                    logger.info("❌ CLIENT DISCONNECTED (/ws)")
                    break

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
                    
        finally:
            # End user session when disconnecting
            if user:
                user.end_session()
                user_manager.save_users()
                user_manager.save_user_to_firebase(user)  # Also save to Firebase
                logger.info(f"👋 User {user.name} session ended")

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
async def process_esp32_audio_save_openai(pcm_bytes: bytes, workflow, websocket):
    """Process ESP32 audio and save OpenAI TTS response to file instead of streaming"""
    try:
        esp32_logger.info(f"🎵 Processing {len(pcm_bytes)} bytes of ESP32 audio")
        
        # Analyze audio quality first
        quality_metrics = analyze_esp32_audio_quality(pcm_bytes)
        esp32_logger.info(f"📊 Audio Quality: {quality_metrics.get('quality', 'unknown')} "
                         f"(RMS: {quality_metrics.get('rms_level', 0):.3f}, "
                         f"Peak: {quality_metrics.get('peak_level', 0):.3f}, "
                         f"SNR: {quality_metrics.get('snr_estimate_db', 0):.1f}dB)")
        
        if quality_metrics.get('is_mostly_silence'):
            esp32_logger.warning("🔇 Audio is mostly silence - skipping processing")
            await websocket.send_bytes(b"EOF")
            return
        
        # Convert PCM bytes to voice pipeline format
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        
        # Apply input preprocessing to reduce noise and DC offset
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        # Remove DC offset
        dc_offset = np.mean(audio_float32)
        audio_float32 = audio_float32 - dc_offset
        if abs(dc_offset) > 0.01:
            esp32_logger.info(f"🔧 Removed input DC offset: {dc_offset:.4f}")
        
        # Apply gentle noise reduction and input filtering for better transcription
        if len(audio_float32) > 100:
            # Enhanced noise gate for better speech clarity
            noise_threshold = 0.003
            noise_gate_ratio = 0.05
            mask = np.abs(audio_float32) < noise_threshold
            audio_float32[mask] *= noise_gate_ratio
            
            # Apply band-pass filter optimized for human speech (300Hz - 3400Hz)
            from scipy import signal
            # High-pass filter to remove low-frequency noise
            sos_hp = signal.butter(2, 300, btype='high', fs=8000, output='sos')
            audio_float32 = signal.sosfilt(sos_hp, audio_float32)
            
            # Low-pass filter to remove high-frequency noise while preserving speech
            sos_lp = signal.butter(2, 3400, btype='low', fs=8000, output='sos')
            audio_float32 = signal.sosfilt(sos_lp, audio_float32)
            
            # Ensure the result is still a numpy array of float32
            audio_float32 = np.array(audio_float32, dtype=np.float32)
            
            esp32_logger.debug("🎤 Applied speech-optimized filtering for better transcription")
        
        # Normalize input audio for better transcription accuracy
        max_val = np.max(np.abs(audio_float32))
        if max_val > 0:
            target_level = 0.7
            audio_float32 = audio_float32 / max_val * target_level
            esp32_logger.debug(f"🔊 Normalized input audio for transcription (max: {max_val:.3f} -> {target_level})")
        
        # Ensure audio_float32 is definitely a numpy array of float32 before creating AudioInput
        audio_float32 = np.asarray(audio_float32, dtype=np.float32)
        audio_input = AudioInput(audio_float32)
        
        esp32_logger.info("🎙️ Starting voice pipeline processing...")
        
        start_time = time.perf_counter()
        
        # Use larger buffer for complete audio generation
        output = await VoicePipeline(
            workflow=workflow,
            config=VoicePipelineConfig(
                tts_settings=TTSModelSettings(
                    buffer_size=4096,
                )
            )
        ).run(audio_input)
        
        esp32_logger.info("🔊 Waiting for OpenAI TTS audio generation...")
        
        # COLLECT ALL AUDIO CHUNKS FROM OPENAI
        all_audio_chunks = []
        chunk_count = 0
        
        async for event in output.stream():
            if isinstance(event, VoiceStreamEventAudio):
                all_audio_chunks.append(event.data)
                chunk_count += 1
                
                # Log progress
                if chunk_count % 5 == 0:
                    esp32_logger.debug(f"📥 Collected {chunk_count} audio chunks from OpenAI TTS...")
        
        # Log completion
        total_time = time.perf_counter() - start_time
        esp32_logger.info(f"✅ OpenAI TTS generation finished! {chunk_count} chunks in {total_time:.2f}s")
        
        if not all_audio_chunks:
            esp32_logger.warning("⚠️ No audio chunks received from TTS")
            await websocket.send_bytes(b"EOF")
            return
        
        # Concatenate all audio into one complete stream
        esp32_logger.info("🔗 Creating single continuous audio stream...")
        complete_audio = np.concatenate(all_audio_chunks)
        esp32_logger.info(f"🎵 Complete audio ready: {len(complete_audio)} samples ({len(complete_audio)/24000:.1f} seconds)")
        
        # SAVE OPENAI AUDIO TO FILE
        # Create audio directory if it doesn't exist
        audio_dir = "openai_audio_files"
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
            esp32_logger.info(f"📁 Created directory: {audio_dir}")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        filename = os.path.join(audio_dir, f"openai_audio_{timestamp}.wav")
        
        # Save as WAV file (24kHz, 16-bit, mono)
        # OpenAI TTS typically outputs at 24kHz
        sample_rate = 24000
        
        # Convert float32 to int16 for WAV file
        if complete_audio.dtype == np.float32 or complete_audio.dtype == np.float64:
            audio_int16 = (np.clip(complete_audio, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            audio_int16 = complete_audio.astype(np.int16)
        
        # Write WAV file
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)   # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        esp32_logger.info(f"💾 SAVED OpenAI audio to: {filename}")
        esp32_logger.info(f"📊 File details: {len(audio_int16)} samples, {len(audio_int16)/sample_rate:.2f} seconds, {sample_rate}Hz")
        
        # Also save in other formats for flexibility
        # Save as raw float32 numpy array
        np_filename = os.path.join(audio_dir, f"openai_audio_{timestamp}.npy")
        np.save(np_filename, complete_audio)
        esp32_logger.info(f"💾 Also saved as numpy array: {np_filename}")
        
        # Send acknowledgment to ESP32 (not the audio)
        ack_message = f"SAVED:{filename}".encode('utf-8')
        await websocket.send_bytes(ack_message)
        esp32_logger.info(f"📤 Sent acknowledgment to ESP32: {ack_message.decode('utf-8')}")
        
    except Exception as e:
        esp32_logger.error(f"❌ Error in ESP32 audio processing: {e}")
        import traceback
        esp32_logger.error(f"📋 Full traceback: {traceback.format_exc()}")
    
    finally:
        # Always send EOF to signal completion
        try:
            await websocket.send_bytes(b"EOF")
            esp32_logger.info("🏁 EOF signal sent - processing complete")
        except:
            esp32_logger.error("❌ Failed to send EOF signal")

# Update the ESP32 bridge endpoint to use the save function
@app.websocket("/upload")
async def esp32_bridge_endpoint_save_audio(websocket: WebSocket):
    """ESP32 bridge that saves OpenAI audio instead of streaming it back"""
    await websocket.accept()
    esp32_logger.info("🤖 ESP32 CONNECTED to bridge endpoint (SAVE MODE)")
    
    internal_connection = WebsocketHelper(websocket, [], starting_agent)
    audio_buffer = []
    
    workflow = EnhancedWorkflow(internal_connection)
    session_start = time.time()
    audio_chunks_received = 0
    total_bytes_received = 0
    
    try:
        while True:
            data = await websocket.receive_bytes()
            data_size = len(data)
            
            if data == b"NODATA":
                esp32_logger.warning("⚠️ ESP32: No useful audio data received")
                audio_buffer = []
                continue
                
            elif data == b"END":
                session_duration = time.time() - session_start
                esp32_logger.info(f"🎯 ESP32: Audio recording complete! Session: {session_duration:.1f}s, "
                                f"Chunks: {audio_chunks_received}, Bytes: {total_bytes_received}")
                
                if audio_buffer:
                    pcm_bytes = b''.join(audio_buffer)
                    
                    # Use the save function instead of streaming back
                    await process_esp32_audio_save_openai(pcm_bytes, workflow, websocket)
                    
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
                if audio_chunks_received % 200 == 0:
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

# Add a new endpoint to list saved audio files
@app.get("/audio_files")
async def list_audio_files():
    """List all saved OpenAI audio files"""
    audio_dir = "openai_audio_files"
    
    if not os.path.exists(audio_dir):
        return {"files": [], "message": "No audio files directory found"}
    
    files = []
    for filename in sorted(os.listdir(audio_dir), reverse=True):
        if filename.endswith(('.wav', '.npy')):
            filepath = os.path.join(audio_dir, filename)
            file_stats = os.stat(filepath)
            files.append({
                "filename": filename,
                "size_bytes": file_stats.st_size,
                "created": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "type": "WAV" if filename.endswith('.wav') else "NumPy Array"
            })
    
    return {
        "files": files[:50],  # Return last 50 files
        "total_files": len(files),
        "directory": audio_dir
    }

# Add endpoint to serve audio files
@app.get("/audio_files/{filename}")
async def get_audio_file(filename: str):
    """Serve a specific audio file"""
    audio_dir = "openai_audio_files"
    filepath = os.path.join(audio_dir, filename)
    
    if not os.path.exists(filepath):
        return {"error": "File not found"}
    
    if filename.endswith('.wav'):
        return FileResponse(filepath, media_type="audio/wav", filename=filename)
    elif filename.endswith('.npy'):
        return FileResponse(filepath, media_type="application/octet-stream", filename=filename)
    else:
        return {"error": "Unsupported file type"}

# Health check endpoint with logging
@app.get("/health")
async def health_check():
    logger.info("🏥 Health check requested")
    user_stats = user_manager.get_user_stats()
    return {
        "status": "healthy", 
        "endpoints": ["/ws", "/upload"],
        "user_stats": user_stats,
        "conversation_stats": {
            "session_duration": str(datetime.now() - conversation_tracker.session_start),
            "conversation_turns": conversation_tracker.conversation_turns,
            "audio_exchanges": conversation_tracker.audio_exchanges,
            "spanish_words_learned": len(conversation_tracker.spanish_words_learned),
            "characters_met": len(conversation_tracker.characters_met),
            "crystals_found": len(conversation_tracker.crystals_found)
        }
    }

# User management endpoints
@app.post("/register")
async def register_user(registration: UserRegistration):
    """Register a new user with device_id, name, and age"""
    try:
        user = user_manager.register_user(
            device_id=registration.device_id,
            name=registration.name,
            age=registration.age
        )
        
        return {
            "success": True,
            "message": "User registered successfully",
            "user": user.to_dict()
        }
    except Exception as e:
        logger.error(f"❌ Error registering user: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.get("/users")
async def get_all_users():
    """Get all users and their data"""
    return {
        "users": [user.to_dict() for user in user_manager.users.values()],
        "total_users": len(user_manager.users),
        "stats": user_manager.get_user_stats()
    }

@app.get("/users/{user_id}")
async def get_user(user_id: str):
    """Get specific user data"""
    if user_id in user_manager.users:
        return user_manager.users[user_id].to_dict()
    else:
        raise HTTPException(status_code=404, detail="User not found")

@app.get("/users/device/{device_id}")
async def get_user_by_device(device_id: str):
    """Get user data by device ID"""
    if device_id in user_manager.device_to_user:
        user_id = user_manager.device_to_user[device_id]
        return user_manager.users[user_id].to_dict()
    else:
        raise HTTPException(status_code=404, detail="Device not found")

@app.get("/metrics/{device_id}")
async def get_user_metrics(device_id: str):
    """Get comprehensive metrics for a specific device/user"""
    # Check if user exists locally
    if device_id in user_manager.device_to_user:
        user_id = user_manager.device_to_user[device_id]
        user = user_manager.users[user_id]
    else:
        # Try to load from Firebase
        firebase_user = user_manager.load_user_from_firebase(device_id)
        if firebase_user:
            user = firebase_user
        else:
            raise HTTPException(status_code=404, detail="Device not found")
    
    # Calculate additional metrics
    avg_session_time = user.total_time / max(1, len([t for t in user.topics_learnt if t]))  # Rough estimate
    words_per_topic = len(user.words_learnt) / max(1, len(user.topics_learnt))
    learning_velocity = len(user.words_learnt) / max(1, user.total_time / 3600)  # words per hour
    
    return {
        "device_id": user.device_id,
        "user_info": {
            "name": user.name,
            "age": user.age,
            "status": user.status,
            "created_at": user.created_at.isoformat(),
            "last_active": user.last_active.isoformat()
        },
        "learning_progress": {
            "current_episode": user.progress.episode,
            "episodes_completed": user.progress.episodes_completed,
            "current_season": user.progress.season,
            "last_completed_episode": user.last_completed_episode.isoformat() if user.last_completed_episode else None,
            "completion_rate": (user.progress.episodes_completed / max(1, user.progress.episode)) * 100
        },
        "vocabulary_metrics": {
            "total_words_learned": len(user.words_learnt),
            "words_list": user.words_learnt,
            "unique_words": len(set(user.words_learnt)),
            "vocabulary_diversity": len(set(user.words_learnt)) / max(1, len(user.words_learnt)) * 100
        },
        "topic_metrics": {
            "total_topics_covered": len(user.topics_learnt),
            "topics_list": user.topics_learnt,
            "words_per_topic": round(words_per_topic, 2)
        },
        "time_metrics": {
            "total_time_seconds": user.total_time,
            "total_time_minutes": round(user.total_time / 60, 2),
            "total_time_hours": round(user.total_time / 3600, 2),
            "average_session_time_minutes": round(avg_session_time / 60, 2),
            "learning_velocity_words_per_hour": round(learning_velocity, 2)
        },
        "engagement_metrics": {
            "days_since_creation": (datetime.now(timezone.utc) - user.created_at).days,
            "days_since_last_active": (datetime.now(timezone.utc) - user.last_active).days,
            "is_active_user": (datetime.now(timezone.utc) - user.last_active).days <= 7,
            "engagement_score": min(100, (len(user.words_learnt) * 2) + (user.progress.episodes_completed * 10) + (user.total_time / 60))
        },
        "achievement_metrics": {
            "milestones_reached": {
                "first_word": len(user.words_learnt) >= 1,
                "ten_words": len(user.words_learnt) >= 10,
                "fifty_words": len(user.words_learnt) >= 50,
                "first_topic": len(user.topics_learnt) >= 1,
                "five_topics": len(user.topics_learnt) >= 5,
                "first_episode": user.progress.episodes_completed >= 1,
                "five_episodes": user.progress.episodes_completed >= 5,
                "one_hour_learning": user.total_time >= 3600,
                "ten_hours_learning": user.total_time >= 36000
            }
        },
        "raw_data": user.to_dict()
    }

@app.post("/users/{user_id}/name")
async def update_user_name(user_id: str, user_update: UserUpdate):
    """Update user name"""
    if user_id in user_manager.users:
        user_manager.users[user_id].name = user_update.name
        user_manager.save_users()
        user_manager.save_user_to_firebase(user_manager.users[user_id])  # Also update Firebase
        return {"success": True, "name": user_update.name}
    raise HTTPException(status_code=404, detail="User not found")

@app.get("/stats")
async def get_statistics():
    """Get comprehensive statistics"""
    user_stats = user_manager.get_user_stats()
    
    # Top performers
    top_words = sorted(user_manager.users.values(), key=lambda u: len(u.words_learnt), reverse=True)[:5]
    top_time = sorted(user_manager.users.values(), key=lambda u: u.total_time, reverse=True)[:5]
    top_episodes = sorted(user_manager.users.values(), key=lambda u: u.progress.episodes_completed, reverse=True)[:5]
    
    return {
        "overall": user_stats,
        "top_performers": {
            "most_words_learned": [{"name": u.name, "words": len(u.words_learnt)} for u in top_words],
            "most_time_spent": [{"name": u.name, "time": u.total_time} for u in top_time],
            "most_episodes_completed": [{"name": u.name, "episodes": u.progress.episodes_completed} for u in top_episodes]
        },
        "conversation_activity": {
            "session_duration": str(datetime.now() - conversation_tracker.session_start),
            "total_turns": conversation_tracker.conversation_turns,
            "audio_exchanges": conversation_tracker.audio_exchanges
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
            "/register": "POST endpoint to register new users with device_id, name, and age",
            "/health": "Health check with conversation stats",
            "/users": "Get all users data",
            "/users/{user_id}": "Get specific user data",
            "/users/device/{device_id}": "Get user data by device ID",
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
    print("🚀 Starting Enhanced Voice Agent Server with Firebase Integration")
    print("🔥 Firebase Firestore: Connected for user data sync")
    print("📡 ESP32 endpoint: ws://localhost:5001/upload")
    print("🌐 New clients endpoint: ws://localhost:5001/ws")
    print("📝 User registration: POST http://localhost:5001/register")
    print("📊 Health check: http://localhost:5001/health")
    print("👥 Users API: http://localhost:5001/users")
    print("� User metrics: GET http://localhost:5001/metrics/{device_id}")
    print("�📋 Logs endpoint: http://localhost:5001/logs")
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
    print("  • Firebase real-time data sync")
    print("\n📝 User Registration:")
    print("  POST /register with {device_id, name, age}")
    print("  All user data automatically synced to Firebase")
    
    uvicorn.run("server:app", host="0.0.0.0", port=5000, reload=True)
