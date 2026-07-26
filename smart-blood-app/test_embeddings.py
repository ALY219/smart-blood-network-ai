import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Text to turn into numbers
sample_text = "O-negative is the universal red blood cell donor type."

# Generate 768-dimensional vector
response = client.models.embed_content(
    model="text-embedding-004",
    contents=sample_text
)

embedding_vector = response.embedding.values
print(f"Generated Vector Length: {len(embedding_vector)}")
print(f"First 5 dimensions: {embedding_vector[:5]}")