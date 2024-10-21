from groq import Groq

# Store your API key here
API_KEY = "your_Groq_API_key"

# Create Groq client
client = Groq(api_key=API_KEY)

# Llama 3 8B model
MODEL = "llama3-groq-70b-8192-tool-use-preview"
