from groq import Groq

# Store your API key here
API_KEY = "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"

# Create Groq client
client = Groq(api_key=API_KEY)

# Llama 3 8B model
MODEL = "llama3-groq-70b-8192-tool-use-preview"