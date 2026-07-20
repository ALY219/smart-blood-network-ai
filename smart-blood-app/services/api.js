const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL;

/**
 * Sends a message to the FastAPI Gemini agent backend
 * @param {string} userMessage - The text prompt from the user
 * @returns {Promise<object>} The JSON response from the backend agent
 */
export const askBloodAgent = async (userMessage) => {
  try {
    // Making sure the URL is properly formatted
    const response = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: userMessage }),
    });

    if (!response.ok) {
      throw new Error(`Server responded with status: ${response.status}`);
    }

    const data = await response.json();
    return data; 
  } catch (error) {
    console.error("Error communicating with AI agent backend:", error);
    throw error;
  }
};