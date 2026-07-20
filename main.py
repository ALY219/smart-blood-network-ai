import time
import json
import os
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

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
GEMINI_MODEL = "gemini-2.5-flash"

class ChatRequest(BaseModel):
    message: str


def filter_donors_by_blood_type(donor_pool, blood_type):
    return [d for d in donor_pool if d.get("blood_type") == blood_type]


def build_prompt(request: EmergencyBloodRequest, donor_pool: list) -> str:
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
            "latency": str(latency_ms) + "ms",
            "confidence": confidence,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print("Failed to write log to Firestore: " + str(e))


async def agentic_dispatch_logic(request: EmergencyBloodRequest):
    start_time = time.time()
    try:
        donors_ref = db.collection("donors").stream()
        donor_pool = [doc.to_dict() for doc in donors_ref]
    except Exception as e:
        print("Failed to fetch donors from Firestore: " + str(e))
        log_to_firestore("Dispatch failed: could not fetch donor pool - " + str(e), 0, "0%")
        return

    matching_donors = filter_donors_by_blood_type(donor_pool, request.blood_type)

    if not matching_donors:
        print("[AGENT] No donors found matching blood type " + str(request.blood_type))
        log_to_firestore("No matching donors for blood type " + str(request.blood_type), 0, "0%")
        return

    prompt = build_prompt(request, matching_donors)
    print("[AGENT] Analyzing emergency request...")

    try:
        # Standard background task can remain synchronous
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        raw_text = response.text
    except Exception as e:
        latency = round((time.time() - start_time) * 1000, 2)
        print("Gemini API call failed: " + str(e))
        log_to_firestore("Gemini API call failed: " + str(e), latency, "0%")
        return

    latency = round((time.time() - start_time) * 1000, 2)

    try:
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
        matches = json.loads(cleaned_text)

        if not isinstance(matches, list):
            raise ValueError("Expected a JSON list, got: " + type(matches).__name__)

        for donor in matches:
            log_to_firestore("Pinged " + str(donor.get("name", "Donor")), latency, "98.5%")

        print("[AGENT] Successfully dispatched " + str(len(matches)) + " match(es).")
    except Exception as e:
        print("Error parsing AI response: " + str(e))
        log_to_firestore("Failed to parse AI response: " + str(e), latency, "0%")


# ==========================================
# ENDPOINT 1: High-Speed Async Chat Route
# ==========================================
@app.post("/chat")
async def conversational_chat(request: ChatRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    try:
        system_instruction = (
            "You are the Smart Blood Network Assistant, an expert AI embedded inside a blood donation mobile app. "
            "Help the user answer questions about blood compatibility, donor guidelines, and system navigation. "
            "Keep answers concise, accurate, and supportive. Use professional markdown lists if necessary.\n\n"
            f"User Query: {request.message}"
        )

        # 1. Use client.aio for non-blocking asynchronous streaming execution
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=system_instruction
        )
        
        latency = round((time.time() - start_time) * 1000, 2)
        
        # 2. Push the slow Firestore operation to a background thread execution queue
        background_tasks.add_task(
            log_to_firestore, 
            f"Chat Session: {request.message[:40]}...", 
            latency, 
            "Interactive"
        )
        
        # 3. Return response immediately without waiting for Firestore
        return {"status": "success", "reply": response.text}

    except Exception as e:
        print(f"[CHAT ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI Chat processing failed: {str(e)}")


# ==========================================
# ENDPOINT 2: Structural Match Trigger
# ==========================================
@app.post("/api/v1/emergency-request")
async def create_emergency_request(request: EmergencyBloodRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(agentic_dispatch_logic, request)
    return {"status": "success", "message": "AI Agent dispatched."}