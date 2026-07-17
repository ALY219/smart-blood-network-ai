from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_emergency_sms(patient_name: str, blood_group: str, hospital_name: str, donor_name: str) -> str:
    prompt = f"""
    You are an emergency medical dispatcher. Write a highly urgent, professional SMS (under 150 characters) to {donor_name}. 
    {patient_name} critically needs {blood_group} blood at {hospital_name}. 
    Tell them to open the Smart Blood Network app immediately to accept the request.
    Do not use emojis. Keep it extremely urgent but not panic-inducing.
    """
    
    # Bumping the version to the active 2.5 model
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    
    return response.text.strip()