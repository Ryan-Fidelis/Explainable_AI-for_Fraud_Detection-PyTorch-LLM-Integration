import os
import pandas as pd
from google import genai 
from dotenv import load_dotenv

# 1. LOAD ENVIRONMENT VARIABLES & INITIALIZE CLIENT
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "API Key not found! Please check if your '.env' file is in the "
        "exact same folder as this script, and that it is named exactly '.env'."
    )

client = genai.Client(api_key=API_KEY)

# 2. LLM INTEGRATION & PROMPT ENGINEERING
def generate_fraud_explanation(transaction_data_dict):
    """
    Receives data from a transaction flagged as fraud and uses Gemini 
    to generate a contextualized explanation for the fraud analyst.
    """
    
    data_context = "\n".join([f"- {key}: {value}" for key, value in transaction_data_dict.items()])
    
    prompt = f"""
    You are a Senior Corporate Fraud Prevention Analyst. 
    Our neural network model has flagged the transaction below as a FRAUD with high probability.

    TRANSACTION DATA:
    {data_context}

    Your task:
    1. Analyze the features (e.g., high amount z-score, suspicious time, weekend status, etc.).
    2. Explain clearly and directly the likely type of fraud (e.g., Smurfing, Expense Inflation, Out-of-Hours Transaction, Shell Merchant Collusion).
    3. Justify why this transaction is suspicious based exclusively on the provided data.

    Provide a concise, professional response, ideal for quick reading on a financial alerts dashboard.
    """
    
    try:
        chat = client.chats.create(model='gemini-3.6-flash')
        response = chat.send_message(prompt)
        
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {e}"
