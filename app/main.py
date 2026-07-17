import asyncio
import json
import os
import time
from fastapi import FastAPI, BackgroundTasks
from firebase_admin import credentials, firestore, initialize_app
from google import genai
from app.models import EmergencyBloodRequest, DonorProfile
from app.ai_engine import generate_emergency_sms
from dotenv import load_dotenv

load_dotenv()

# --- 1. Infrastructure Setup ---
app = FastAPI(title="Smart Blood Network AI Agent")

# Initialize Firebase Admin
cred = credentials.Certificate("serviceAccountKey.json")
initialize_app(cred)
db = firestore.client()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
# SWITCHED TO 1.5 FLASH TO AVOID 2.0 QUOTA LIMITS
GEMINI_MODEL = "gemini-3.5-flash"

# --- 2. The AI Agentic Dispatcher ---
async def agentic_dispatch_logic(request: EmergencyBloodRequest):
    print("\n[AGENT] Analyzing emergency for " + request.patient_name + "...")

    donors_ref = db.collection("donors").stream()
    donor_pool = [doc.to_dict() for doc in donors_ref]

    prompt = (
        "Emergency Request: " + request.model_dump_json() + "\n" +
        "Available Donors: " + json.dumps(donor_pool) + "\n" +
        "Task: Identify donors who are a blood type match.\n" +
        "Return ONLY a valid JSON list of objects in this exact format: " + '[{"name": "string", "phone": "string"}]' + "\n" +
        "Do not include any text outside the JSON list. Do not use markdown code fences."
    )

    # Added Retry Logic for Quota Limits
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        raw_text = response.text
    except Exception as e:
        if "429" in str(e):
            print("[AGENT] Quota hit. Waiting 60 seconds before retry...")
            await asyncio.sleep(60)
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            raw_text = response.text
        else:
            print("Gemini API call failed: " + str(e))
            return

    try:
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
        matches = json.loads(cleaned_text)
    except Exception as e:
        print("Error parsing AI response: " + str(e))
        return

    for donor in matches:
        donor_name = donor.get("name", "Unknown Donor")
        msg = generate_emergency_sms(
            patient_name=request.patient_name,
            blood_group=request.blood_group,
            hospital_name=request.hospital_name,
            donor_name=donor_name
        )

        print("[AGENT] Pinging " + donor_name + ": " + msg)

        try:
            db.collection("logs").add({
                "donor": donor_name,
                "message": msg,
                "status": "sent"
            })
        except Exception as e:
            print("Failed to write log to Firestore: " + str(e))

@app.post("/api/v1/emergency-request")
async def create_emergency_request(request: EmergencyBloodRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(agentic_dispatch_logic, request)
    return {
        "status": "success",
        "message": "AI Agent is now processing the emergency and matching donors."
    }