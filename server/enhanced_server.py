import os
import asyncio
import json
import uuid
import base64
import io
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import ollama
import httpx
from faster_whisper import WhisperModel

# Pydantic models for API
class PatientCreate(BaseModel):
    name: str
    age: Optional[int] = None
    notes: Optional[str] = ""

class ChatMessage(BaseModel):
    message: str
    patient_id: Optional[str] = None

class DocumentUpload(BaseModel):
    content: str
    filename: str
    patient_id: str

# FastAPI app setup
app = FastAPI(title="Phlox Enhanced")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
BUILD_DIR = Path("../build")
CHROMA_DB_DIR = Path("./chroma_db")

# Initialize ChromaDB
try:
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Create collections
    patients_collection = chroma_client.get_or_create_collection(
        name="patients",
        metadata={"description": "Patient medical records and documents"}
    )
    
    chat_collection = chroma_client.get_or_create_collection(
        name="chat_history", 
        metadata={"description": "Chat conversations for RAG"}
    )
    
    print("✅ ChromaDB initialized successfully")
except Exception as e:
    print(f"❌ Error initializing ChromaDB: {e}")
    chroma_client = None

# Ollama client setup
def get_ollama_response(prompt: str, model: str = "llama3.1:latest") -> str:
    """Get response from local Llama model via Ollama"""
    try:
        response = ollama.chat(model=model, messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']
    except Exception as e:
        print(f"Error with Ollama: {e}")
        return f"Error connecting to Llama model: {str(e)}"

# In-memory storage (replace with database in production)
patients_db: Dict[str, dict] = {}
chat_history: Dict[str, List[dict]] = {}
todos: List[dict] = [
    {"id": "1", "task": "Review patient charts", "completed": False, "priority": "high"},
    {"id": "2", "task": "Update treatment protocols", "completed": False, "priority": "medium"},
    {"id": "3", "task": "Schedule team meeting", "completed": True, "priority": "low"}
]

# Initialize Whisper model
try:
    # Use base model for faster loading, you can change to small, medium, large for better accuracy
    whisper_model = WhisperModel("base", device="auto", compute_type="auto")
    print("✅ Faster-Whisper initialized successfully")
except Exception as e:
    print(f"❌ Error initializing Whisper: {e}")
    whisper_model = None

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok", 
        "message": "Enhanced Phlox server with ChromaDB and Llama", 
        "chromadb": "connected" if chroma_client else "error",
        "ollama": "connected",
        "whisper": "connected" if whisper_model else "error"
    }

# Configuration endpoints
@app.get("/api/config/version")
async def get_version():
    return {"version": "1.0.0-enhanced", "build": "local"}

@app.get("/api/config/changelog")
async def get_changelog():
    return {"changelog": "Enhanced with ChromaDB and local Llama integration"}

@app.get("/api/config/status")
async def get_status():
    return {
        "server": "online",
        "database": "connected" if chroma_client else "error",
        "llm": "ollama-llama3.1",
        "whisper": "connected" if whisper_model else "error",
        "features": ["chromadb", "local-llm", "rag", "whisper-transcription"]
    }

@app.get("/api/config/global")
async def get_global_config():
    """Get global configuration"""
    return {
        "app_name": "Phlox Enhanced",
        "version": "1.0.0-enhanced",
        "features": ["chromadb", "local-llm", "rag"],
        "ollama_model": "llama3.1:latest"
    }

@app.get("/api/config/ollama")
async def get_ollama_config():
    """Get Ollama configuration"""
    try:
        # Check if Ollama is running and get model info
        response = ollama.list()
        if response and 'models' in response:
            return {
                "status": "connected",
                "models": response['models'],
                "default_model": "llama3.1:latest"
            }
        else:
            return {"status": "disconnected", "models": [], "default_model": None}
    except Exception as e:
        return {"status": "error", "error": str(e), "models": [], "default_model": None}

@app.get("/api/config/user")
async def get_user_config():
    """Get user configuration"""
    return {
        "user_id": "default",
        "preferences": {
            "theme": "dark",
            "auto_save": True,
            "notifications": True
        }
    }

@app.get("/api/config/whisper")
async def get_whisper_config():
    """Get Whisper configuration and status"""
    if not whisper_model:
        return {
            "status": "error",
            "model": None,
            "error": "Whisper model not initialized"
        }
    
    return {
        "status": "connected",
        "model": "faster-whisper base",
        "device": "auto",
        "compute_type": "auto"
    }

# Patient endpoints
@app.get("/api/patient/list")
async def get_patient_list(date: Optional[str] = None):
    """Get list of patients for a given date"""
    # Convert patients_db to list format expected by frontend
    patient_list = []
    for patient_id, patient_data in patients_db.items():
        patient_list.append({
            "id": patient_id,
            "name": patient_data.get("name", "Unknown"),
            "time": patient_data.get("time", "09:00"),
            "status": patient_data.get("status", "scheduled"),
            "age": patient_data.get("age"),
            "notes": patient_data.get("notes", "")
        })
    
    return {"patients": patient_list}

@app.get("/api/patient/incomplete-jobs-count")
async def get_incomplete_jobs_count():
    """Get count of incomplete jobs"""
    return {"count": 2}

@app.get("/api/patient/outstanding-jobs") 
async def get_outstanding_jobs():
    """Get outstanding patient jobs"""
    return {"jobs": []}

@app.post("/api/patient/create")
async def create_patient(patient: PatientCreate):
    """Create a new patient"""
    patient_id = str(uuid.uuid4())
    patient_data = {
        "id": patient_id,
        "name": patient.name,
        "age": patient.age,
        "notes": patient.notes,
        "created_at": datetime.now().isoformat(),
        "status": "scheduled",
        "time": "09:00"
    }
    
    patients_db[patient_id] = patient_data
    
    # Store in ChromaDB if available
    if chroma_client and patients_collection:
        try:
            patients_collection.add(
                documents=[f"Patient: {patient.name}, Age: {patient.age}, Notes: {patient.notes}"],
                metadatas=[{"patient_id": patient_id, "type": "patient_record"}],
                ids=[patient_id]
            )
        except Exception as e:
            print(f"Error storing in ChromaDB: {e}")
    
    return {"success": True, "patient_id": patient_id, "patient": patient_data}

@app.get("/api/patient/{patient_id}")
async def get_patient(patient_id: str):
    """Get specific patient details"""
    if patient_id in patients_db:
        return patients_db[patient_id]
    raise HTTPException(status_code=404, detail="Patient not found")

# Chat endpoints
@app.post("/api/chat")
async def chat_with_llama(message: ChatMessage):
    """Chat with local Llama model with RAG"""
    try:
        # Get relevant context from ChromaDB if available
        context = ""
        if chroma_client and patients_collection:
            try:
                # Query for relevant patient information
                results = patients_collection.query(
                    query_texts=[message.message],
                    n_results=3
                )
                if results['documents'] and results['documents'][0]:
                    context = "Relevant patient information:\n" + "\n".join(results['documents'][0])
            except Exception as e:
                print(f"Error querying ChromaDB: {e}")
        
        # Construct prompt with context
        prompt = f"""You are a medical AI assistant. Answer the following question based on the context provided.

Context:
{context}

Question: {message.message}

Please provide a helpful and accurate response:"""
        
        # Get response from Llama
        response = get_ollama_response(prompt)
        
        # Store conversation in chat history
        chat_id = message.patient_id or "general"
        if chat_id not in chat_history:
            chat_history[chat_id] = []
        
        conversation = {
            "id": str(uuid.uuid4()),
            "user_message": message.message,
            "assistant_response": response,
            "timestamp": datetime.now().isoformat()
        }
        
        chat_history[chat_id].append(conversation)
        
        # Store in ChromaDB for future RAG
        if chroma_client and chat_collection:
            try:
                chat_collection.add(
                    documents=[f"User: {message.message}\nAssistant: {response}"],
                    metadatas=[{"patient_id": chat_id, "type": "conversation"}],
                    ids=[conversation["id"]]
                )
            except Exception as e:
                print(f"Error storing chat in ChromaDB: {e}")
        
        return {
            "response": response,
            "conversation_id": conversation["id"],
            "context_used": bool(context)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/api/chat/history/{chat_id}")
async def get_chat_history(chat_id: str):
    """Get chat history for a patient or general chat"""
    return {"history": chat_history.get(chat_id, [])}

# Document upload and processing
@app.post("/api/documents/upload")
async def upload_document(doc: DocumentUpload):
    """Upload and process document for RAG"""
    try:
        if not chroma_client:
            raise HTTPException(status_code=500, detail="ChromaDB not available")
        
        # Process and chunk the document (simplified)
        doc_id = str(uuid.uuid4())
        
        # Store in ChromaDB
        patients_collection.add(
            documents=[doc.content],
            metadatas=[{
                "patient_id": doc.patient_id,
                "filename": doc.filename,
                "type": "document",
                "uploaded_at": datetime.now().isoformat()
            }],
            ids=[doc_id]
        )
        
        return {"success": True, "document_id": doc_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

# Dashboard endpoints
@app.get("/api/dashboard/todos")
async def get_todos():
    """Get dashboard todos"""
    return {"todos": todos}

@app.get("/api/dashboard/rss/list")
async def get_rss_list():
    return {"feeds": []}

@app.get("/api/dashboard/rss/digest")
async def get_rss_digest():
    return {"articles": []}

@app.get("/api/dashboard/analysis/latest")
async def get_latest_analysis():
    return {
        "analysis": "System running optimally with ChromaDB and local Llama integration",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/dashboard/server/info")
async def get_server_info():
    return {
        "server": "Phlox Enhanced",
        "version": "1.0.0",
        "features": ["ChromaDB", "Local Llama", "RAG"],
        "uptime": "Active",
        "database": "ChromaDB Connected" if chroma_client else "ChromaDB Error"
    }

# Templates endpoints
@app.get("/api/templates")
async def get_templates():
    """Get all templates with correct structure for frontend"""
    # Return proper template structure that matches frontend expectations
    mock_templates = [
        {
            "template_key": "phlox_general_v1",
            "template_name": "General Consultation",
            "fields": [
                {
                    "field_key": "chief_complaint",
                    "field_name": "Chief Complaint",
                    "field_type": "text",
                    "required": True,
                    "persistent": False,
                    "system_prompt": "Extract the patient's main concern",
                    "initial_prompt": "What brings you in today?",
                    "format_schema": "",
                    "refinement_rules": ""
                },
                {
                    "field_key": "history_present_illness",
                    "field_name": "History of Present Illness",
                    "field_type": "text",
                    "required": True,
                    "persistent": False,
                    "system_prompt": "Detail the current symptoms and timeline",
                    "initial_prompt": "Tell me more about your symptoms",
                    "format_schema": "",
                    "refinement_rules": ""
                },
                {
                    "field_key": "assessment",
                    "field_name": "Assessment",
                    "field_type": "text",
                    "required": True,
                    "persistent": False,
                    "system_prompt": "Clinical assessment and diagnosis",
                    "initial_prompt": "Based on the information provided",
                    "format_schema": "",
                    "refinement_rules": ""
                },
                {
                    "field_key": "plan",
                    "field_name": "Plan",
                    "field_type": "text",
                    "required": True,
                    "persistent": False,
                    "system_prompt": "Treatment plan and follow-up",
                    "initial_prompt": "Recommended treatment plan",
                    "format_schema": "",
                    "refinement_rules": ""
                }
            ]
        },
        {
            "template_key": "soap_v1",
            "template_name": "SOAP Note",
            "fields": [
                {
                    "field_key": "subjective",
                    "field_name": "Subjective",
                    "field_type": "text",
                    "required": True,
                    "persistent": False,
                    "system_prompt": "Patient-reported symptoms and concerns",
                    "initial_prompt": "Patient reports",
                    "format_schema": "",
                    "refinement_rules": ""
                },
                {
                    "field_key": "objective",
                    "field_name": "Objective",
                    "field_type": "text",
                    "required": True,
                    "persistent": False,
                    "system_prompt": "Observable findings and measurements",
                    "initial_prompt": "On examination",
                    "format_schema": "",
                    "refinement_rules": ""
                },
                {
                    "field_key": "assessment",
                    "field_name": "Assessment",
                    "field_type": "text",
                    "required": True,
                    "persistent": False,
                    "system_prompt": "Clinical assessment and diagnosis",
                    "initial_prompt": "Assessment",
                    "format_schema": "",
                    "refinement_rules": ""
                },
                {
                    "field_key": "plan",
                    "field_name": "Plan",
                    "field_type": "text",
                    "required": True,
                    "persistent": False,
                    "system_prompt": "Treatment plan and follow-up",
                    "initial_prompt": "Plan",
                    "format_schema": "",
                    "refinement_rules": ""
                }
            ]
        }
    ]
    return mock_templates

@app.get("/api/templates/default")
async def get_default_template():
    """Get default template with correct structure"""
    return {
        "template_key": "phlox_general_v1",
        "template_name": "General Consultation",
        "is_default": True,
        "fields": [
            {
                "field_key": "chief_complaint",
                "field_name": "Chief Complaint",
                "field_type": "text",
                "required": True,
                "persistent": False,
                "system_prompt": "Extract the patient's main concern",
                "initial_prompt": "What brings you in today?",
                "format_schema": "",
                "refinement_rules": ""
            },
            {
                "field_key": "history_present_illness",
                "field_name": "History of Present Illness",
                "field_type": "text",
                "required": True,
                "persistent": False,
                "system_prompt": "Detail the current symptoms and timeline",
                "initial_prompt": "Tell me more about your symptoms",
                "format_schema": "",
                "refinement_rules": ""
            },
            {
                "field_key": "assessment",
                "field_name": "Assessment",
                "field_type": "text",
                "required": True,
                "persistent": False,
                "system_prompt": "Clinical assessment and diagnosis",
                "initial_prompt": "Based on the information provided",
                "format_schema": "",
                "refinement_rules": ""
            },
            {
                "field_key": "plan",
                "field_name": "Plan",
                "field_type": "text",
                "required": True,
                "persistent": False,
                "system_prompt": "Treatment plan and follow-up",
                "initial_prompt": "Recommended treatment plan",
                "format_schema": "",
                "refinement_rules": ""
            }
        ]
    }

@app.get("/api/templates/{template_key}")
async def get_template_by_key(template_key: str):
    """Get specific template by key"""
    if template_key == "undefined" or not template_key:
        raise HTTPException(status_code=400, detail="Invalid template key")
    
    # Mock templates database
    templates_db = {
        "phlox_general_v1": {
            "template_key": "phlox_general_v1",
            "template_name": "General Consultation",
            "fields": [
                {
                    "field_key": "chief_complaint",
                    "field_name": "Chief Complaint",
                    "field_type": "text",
                    "required": True,
                    "persistent": False,
                    "system_prompt": "Extract the patient's main concern",
                    "initial_prompt": "What brings you in today?",
                    "format_schema": "",
                    "refinement_rules": ""
                },
                {
                    "field_key": "history_present_illness",
                    "field_name": "History of Present Illness",
                    "field_type": "text",
                    "required": True,
                    "persistent": False,
                    "system_prompt": "Detail the current symptoms and timeline",
                    "initial_prompt": "Tell me more about your symptoms",
                    "format_schema": "",
                    "refinement_rules": ""
                },
                {
                    "field_key": "assessment",
                    "field_name": "Assessment",
                    "field_type": "text",
                    "required": True,
                    "persistent": False,
                    "system_prompt": "Clinical assessment and diagnosis",
                    "initial_prompt": "Based on the information provided",
                    "format_schema": "",
                    "refinement_rules": ""
                },
                {
                    "field_key": "plan",
                    "field_name": "Plan",
                    "field_type": "text",
                    "required": True,
                    "persistent": False,
                    "system_prompt": "Treatment plan and follow-up",
                    "initial_prompt": "Recommended treatment plan",
                    "format_schema": "",
                    "refinement_rules": ""
                }
            ]
        },
        "soap_v1": {
            "template_key": "soap_v1",
            "template_name": "SOAP Note",
            "fields": [
                {
                    "field_key": "subjective",
                    "field_name": "Subjective",
                    "field_type": "text",
                    "required": True,
                    "persistent": False,
                    "system_prompt": "Patient-reported symptoms and concerns",
                    "initial_prompt": "Patient reports",
                    "format_schema": "",
                    "refinement_rules": ""
                },
                {
                    "field_key": "objective",
                    "field_name": "Objective",
                    "field_type": "text",
                    "required": True,
                    "persistent": False,
                    "system_prompt": "Observable findings and measurements",
                    "initial_prompt": "On examination",
                    "format_schema": "",
                    "refinement_rules": ""
                },
                {
                    "field_key": "assessment",
                    "field_name": "Assessment",
                    "field_type": "text",
                    "required": True,
                    "persistent": False,
                    "system_prompt": "Clinical assessment and diagnosis",
                    "initial_prompt": "Assessment",
                    "format_schema": "",
                    "refinement_rules": ""
                },
                {
                    "field_key": "plan",
                    "field_name": "Plan",
                    "field_type": "text",
                    "required": True,
                    "persistent": False,
                    "system_prompt": "Treatment plan and follow-up",
                    "initial_prompt": "Plan",
                    "format_schema": "",
                    "refinement_rules": ""
                }
            ]
        }
    }
    
    if template_key not in templates_db:
        raise HTTPException(status_code=404, detail=f"Template {template_key} not found")
    
    return templates_db[template_key]

# RAG Search endpoint
@app.post("/api/rag/search")
async def rag_search(query: dict):
    """Search using RAG with ChromaDB and Llama"""
    try:
        search_query = query.get("query", "")
        
        if not chroma_client:
            raise HTTPException(status_code=500, detail="ChromaDB not available")
        
        # Search in both collections
        patient_results = patients_collection.query(
            query_texts=[search_query],
            n_results=5
        )
        
        chat_results = chat_collection.query(
            query_texts=[search_query], 
            n_results=5
        )
        
        # Combine results
        all_docs = []
        if patient_results['documents']:
            all_docs.extend(patient_results['documents'][0])
        if chat_results['documents']:
            all_docs.extend(chat_results['documents'][0])
        
        # Generate response using Llama with context
        context = "\n\n".join(all_docs[:3])  # Use top 3 results
        prompt = f"""Based on the following medical context, answer this question: {search_query}

Context:
{context}

Please provide a comprehensive answer:"""
        
        response = get_ollama_response(prompt)
        
        return {
            "answer": response,
            "sources": len(all_docs),
            "context_used": bool(context)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG search error: {str(e)}")

# RAG endpoints  
@app.get("/api/rag/files")
async def get_rag_files():
    """Get list of RAG files"""
    try:
        # Get collection info from ChromaDB
        collections = chroma_client.list_collections()
        files = []
        
        for collection in collections:
            # Get some basic info about each collection
            count = collection.count()
            files.append({
                "id": collection.name,
                "name": collection.name,
                "type": "collection", 
                "size": count,
                "status": "indexed"
            })
        
        return {"files": files}
    except Exception as e:
        return {"files": [], "error": str(e)}

@app.post("/api/rag/upload")
async def upload_rag_file(file_data: dict):
    """Upload and process file for RAG"""
    return {
        "message": "File upload not yet implemented",
        "status": "placeholder"
    }

# Transcription endpoints
@app.post("/api/transcribe/audio")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio using local Faster-Whisper model"""
    if not whisper_model:
        raise HTTPException(status_code=500, detail="Whisper model not available")
    
    try:
        # Read the uploaded audio file
        audio_content = await file.read()
        
        # Save temporarily for processing
        temp_file = f"/tmp/audio_{uuid.uuid4()}.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_content)
        
        # Transcribe with Whisper
        segments, info = whisper_model.transcribe(temp_file, beam_size=5)
        
        # Combine all segments into full text
        transcription = ""
        segments_list = []
        
        for segment in segments:
            transcription += segment.text + " "
            segments_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
        
        # Clean up temp file
        os.remove(temp_file)
        
        return {
            "transcription": transcription.strip(),
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": segments_list,
            "status": "success"
        }
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@app.post("/api/transcribe/audio-base64")
async def transcribe_audio_base64(request: dict):
    """Transcribe audio from base64 data"""
    if not whisper_model:
        raise HTTPException(status_code=500, detail="Whisper model not available")
    
    try:
        # Decode base64 audio data
        audio_data = base64.b64decode(request.get("audio_data", ""))
        
        # Save temporarily for processing
        temp_file = f"/tmp/audio_{uuid.uuid4()}.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_data)
        
        # Transcribe with Whisper
        segments, info = whisper_model.transcribe(temp_file, beam_size=5)
        
        # Combine all segments into full text
        transcription = ""
        segments_list = []
        
        for segment in segments:
            transcription += segment.text + " "
            segments_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
        
        # Clean up temp file
        os.remove(temp_file)
        
        return {
            "transcription": transcription.strip(),
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": segments_list,
            "status": "success"
        }
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

# React app routes
@app.get("/new-patient")
@app.get("/settings") 
@app.get("/rag")
@app.get("/clinic-summary")
@app.get("/outstanding-tasks")
async def serve_react_app():
    return FileResponse(BUILD_DIR / "index.html")

# Serve static files
if (BUILD_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=BUILD_DIR / "static"), name="static")

# Catch-all route for any other paths
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    # Don't intercept API calls
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    
    # If the path is a file that exists, serve it
    file_path = BUILD_DIR / full_path
    if file_path.is_file():
        return FileResponse(file_path)
    # Otherwise serve the React app
    return FileResponse(BUILD_DIR / "index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting Enhanced Phlox server on port {port}")
    print(f"📁 React build directory: {BUILD_DIR.resolve()}")
    print(f"🗄️  ChromaDB directory: {CHROMA_DB_DIR.resolve()}")
    print(f"🤖 Llama model: llama3.1:latest")
    print(f"✨ Features: ChromaDB, Local Llama, RAG, Vector Search")
    
    # Create ChromaDB directory if it doesn't exist
    CHROMA_DB_DIR.mkdir(exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=port) 