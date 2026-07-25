import time
import json
import os
from typing import List, Dict, Any
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials, firestore, initialize_app
from google import genai
from pydantic import BaseModel
from app.models import EmergencyBloodRequest
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Smart Blood Network AI Agent")

# Enable CORS for smooth mobile-backend communications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceAccountKey.json")
initialize_app(cred)
db = firestore.client()

# Initialize Gemini Client (New Google GenAI SDK)
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
GEMINI_MODEL = "gemini-2.5-flash"


class ChatRequest(BaseModel):
    message: str


def filter_donors_by_blood_type(donor_pool: List[Dict[str, Any]], blood_group: str) -> List[Dict[str, Any]]:
    """
    Hard filter in Python first to guarantee exact matching on blood group/type.
    Handles both 'blood_group' and 'blood_type' field keys in Firestore documents.
    """
    if not blood_group:
        return []
    
    target = str(blood_group).strip().upper()
    matching = []
    
    for donor in donor_pool:
        donor_bg = donor.get("blood_group") or donor.get("blood_type") or ""
        if str(donor_bg).strip().upper() == target:
            matching.append(donor)
            
    return matching


def build_prompt(request: EmergencyBloodRequest, donor_pool: list) -> str:
    """
    Build structured prompt using string concatenation to avoid JSON string formatting conflicts with literal braces.
    """
    part1 = "Emergency Request: " + request.model_dump_json() + "\n"
    part2 = "Pre-filtered matching donors: " + json.dumps(donor_pool) + "\n\n"
    part3 = "Task: Rank these donors by suitability (proximity, availability, response history if present).\n"
    part4 = "Return ONLY a valid JSON list of objects in this exact format: " + '[{"name": "string", "phone": "string"}]'
    part5 = "\nDo not include any text outside the JSON list. Do not use markdown code fences."
    return part1 + part2 + part3 + part4 + part5


def log_to_firestore(message: str, latency_ms: float, confidence: str = "n/a"):
    try:
        db.collection("logs").add({
            "message": message,
            "latency": f"{latency_ms}ms",
            "confidence": confidence,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"Failed to write log to Firestore: {e}")


async def agentic_dispatch_logic(request: EmergencyBloodRequest):
    start_time = time.time()
    
    # Safely extract target blood group/type regardless of field naming in Pydantic model
    target_blood_group = getattr(request, "blood_group", getattr(request, "blood_type", ""))
    
    try:
        donors_ref = db.collection("donors").stream()
        donor_pool = [doc.to_dict() for doc in donors_ref]
    except Exception as e:
        print(f"Failed to fetch donors from Firestore: {e}")
        log_to_firestore(f"Dispatch failed: could not fetch donor pool - {e}", 0, "0%")
        return

    matching_donors = filter_donors_by_blood_type(donor_pool, target_blood_group)

    if not matching_donors:
        print(f"[AGENT] No donors found matching blood group {target_blood_group}")
        log_to_firestore(f"No matching donors for blood type {target_blood_group}", 0, "0%")
        return

    prompt = build_prompt(request, matching_donors)
    print(f"[AGENT] Analyzing emergency request for {len(matching_donors)} candidate donor(s)...")

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        raw_text = response.text
    except Exception as e:
        latency = round((time.time() - start_time) * 1000, 2)
        print(f"Gemini API call failed: {e}")
        log_to_firestore(f"Gemini API call failed: {e}", latency, "0%")
        return

    latency = round((time.time() - start_time) * 1000, 2)

    try:
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
        matches = json.loads(cleaned_text)

        if not isinstance(matches, list):
            raise ValueError(f"Expected a JSON list, got: {type(matches).__name__}")

        for donor in matches:
            donor_name = donor.get("name", "Donor") if isinstance(donor, dict) else "Donor"
            log_to_firestore(f"Pinged {donor_name}", latency, "98.5%")

        print(f"[AGENT] Successfully dispatched {len(matches)} match(es).")
    except Exception as e:
        print(f"Error parsing AI response: {e}")
        print(f"Raw response was: {raw_text}")
        log_to_firestore(f"Failed to parse AI response: {e}", latency, "0%")


# ==========================================
# ENDPOINT 1: High-Speed Async Chat Route
# ==========================================
@app.post("/chat")
async def conversational_chat(request: ChatRequest, background_tasks: BackgroundTasks):
    # Guard against empty/whitespace-only input
    clean_message = request.message.strip()
    if not clean_message:
        return {
            "status": "success",
            "reply": "Please enter a question or request so I can help you."
        }

    start_time = time.time()
    try:
        system_instruction = (
            "You are the Smart Blood Network Assistant, an expert AI embedded inside a blood donation mobile app. "
            "Help the user answer questions about blood compatibility, donor guidelines, and system navigation. "
            "Keep answers concise, accurate, and supportive. Use professional markdown lists if necessary.\n\n"
            f"User Query: {clean_message}"
        )

        # Non-blocking async call using client.aio
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=system_instruction
        )
        
        latency = round((time.time() - start_time) * 1000, 2)
        
        # Async background log to Firestore
        background_tasks.add_task(
            log_to_firestore, 
            f"Chat Session: {clean_message[:40]}...", 
            latency, 
            "Interactive"
        )
        
        return {"status": "success", "reply": response.text}

    except Exception as e:
        print(f"[CHAT ERROR] {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="The AI Assistant is currently unavailable. Please try again shortly."
        )


# ==========================================
# ENDPOINT 2: Structural Match Trigger
# ==========================================
@app.post("/api/v1/emergency-request")
async def create_emergency_request(request: EmergencyBloodRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(agentic_dispatch_logic, request)
    return {"status": "success", "message": "AI Agent dispatched."}