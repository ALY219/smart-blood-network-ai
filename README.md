# 🚀 Smart Blood Network: Agentic AI Dispatcher

A high-performance, full-stack, real-time emergency dispatch system. This platform automates the matching of critical blood requests with nearby donors using autonomous AI agents, ensuring rapid response times in life-critical scenarios.

---

### 🏗️ System Architecture

The application utilizes an event-driven architecture to bridge real-time mobile interactions with autonomous backend intelligence:

*   **Frontend (React Native + Expo):** A responsive mobile interface for emergency personnel. It maintains a reactive connection to the database, ensuring that dispatch logs and donor statuses are reflected in real-time.
*   **Backend (FastAPI):** An asynchronous API gateway that manages traffic and offloads AI reasoning to background tasks, keeping the system responsive under load.
*   **Intelligence Layer (Gemini 1.5 Flash):** An agentic engine that ingests emergency metadata and the current donor pool. It performs semantic reasoning to evaluate blood compatibility, proximity, and availability, returning structured match data.
*   **Data Backbone (Firebase Firestore):** A real-time NoSQL database that serves as the "source of truth," synchronizing the state between the Python backend and the React Native frontend.

---

### 🛠️ Key Technical Features

*   **Autonomous AI Reasoning:** Utilizes the Gemini 1.5 Flash model to intelligently process and match donor resources, reducing human error in emergency dispatch.
*   **Asynchronous Background Processing:** Employs FastAPI `BackgroundTasks` to handle intensive AI reasoning without blocking UI interactions.
*   **Reactive UI:** Implements real-time listeners via Firebase `onSnapshot`, providing an instant feedback loop for AI dispatch actions.
*   **Observability Metrics:** Includes automated tracking of AI inference latency and match confidence scores, exposing the "black box" of AI to the end-user for transparency.
*   **Security First:** Strictly adheres to credential management best practices using `.env` files and `python-dotenv`, ensuring that sensitive API keys and service accounts are never exposed in source control.

---

### 🚀 Tech Stack

*   **Language:** Python 3.10+, JavaScript (ES6+)
*   **Frameworks:** FastAPI, React Native (Expo)
*   **AI/LLM:** Google Gemini 1.5 Flash
*   **Database:** Firebase Firestore
*   **Tooling:** `python-dotenv`, Uvicorn, Firebase Admin SDK

---

### 📦 Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ALY219/smart-blood-network-ai.git](https://github.com/ALY219/smart-blood-network-ai.git)
   cd smart-blood-network-ai
