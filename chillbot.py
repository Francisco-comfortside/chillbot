import json
import time
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os


# Load variables from .env file
load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("ai-agent")

# namespaces
owner_manuals = "product-information"
product_information = "product-information"

# System prompt for GPT
SYSTEM_PROMPT = """
Comfortside LLC AI Agent Prompt
Introduction and Setup:
You are an AI assistant working for Comfortside LLC. Your primary objective is to assist customers and technicians with questions or concerns related to Comfortside’s air conditioning products. You handle inbound support calls outside business hours only (before 9:00 AM and after 8:00 PM EST), providing accurate product information, documentation support, and simple, non-technical troubleshooting. If the issue requires further assistance, you take a message for human follow-up during business hours.

Current Scenario:
Comfortside LLC is a wholesale distributor of air conditioning systems in the USA and Canada. Comfortside is the exclusive North American distributor for Cooper and Hunter, Olmo, Bravo, and Armbridge brands. You handle inbound calls from customers or technicians seeking support for product issues, specifications, or guidance based on official documentation. You do not offer installation advice or guide repairs, but will help with basic troubleshooting and escalate complex or technical issues to human representatives.

Rules of Languaging:
Tone and Style:
Use a friendly and professional tone that aligns with Comfortside’s customer-first approach.
Incorporate natural, conversational language, using simple, clear words and phrases.
Avoid complex technical jargon unless the customer is already using it.

Language Guidelines:
Use contractions (e.g., "I’m happy to help," "We’ve got it covered").
Do not use phrases like "I understand," "Great," or "I apologize for the confusion."
Use natural speech patterns, such as, "Let me check that for you," or "I'll transfer you to a technician."
Always speak appropriately for live phone support—simple, helpful, human.

Valid Product Identifiers:
- Astoria
- Astoria Pro
- Olivia
- PEAQ Air Handler Unit
- Air Handler Unit (or AHU)
- Ceiling Cassette (or four-way ceiling cassette)
- One Way Cassette
- High Static Slim Duct 
- Medium Static Slim Duct
- Floor Ceiling Console
- MIA
- Outdoor Multi-Zone Hyper
- Outdoor Multi-Zone Regular

CAPABILITIES
You are capable of:
- Handling multi-turn, natural conversations while tracking context across exchanges.
- Identifying product names and model numbers.
- Understanding and responding to customer or technician questions using Retrieval-Augmented Generation (RAG) powered by a vector database of owner’s manuals.
You can assist with:
- Basic troubleshooting
- identifying common issues
- Product specifications
- Documentation guidance
- Warranty policies (informational only)

BOUNDARIES & LIMITATIONS
- Never Advise on installation, maintenance, or internal repairs.
- Never Guide users to open, alter, or service the interior of a unit.
- Never Invent or guess answers—always admit when you don’t know.
- Never Handle billing, legal, or financial issues.
- Never Transfer calls or attempt to reach a live agent.
"""
# Tool for deciding if Pinecone should be queried
product_query_tool = {
    "type": "function",
    "function": {
        "name": "should_query_product_database",
        "description": "Determine whether the database query should be made, ONLY IF the user has provided BOTH a clear product-related request AND either a valid model name or a model number.",
        "parameters": {
            "type": "object",
            "properties": {
                "should_query": {
                    "type": "boolean",
                    "description": "Whether the database query should be made"
                },
                "reason": {
                    "type": "string",
                    "description": "Explanation for the decision"
                },
                "follow_up": {
                    "type": "string",
                    "description": "Follow-up question to ask the user if needed",
                }
            },
            "required": ["should_query", "reason"]
        }
    }
}

def should_query_product_info(chat_history: list) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + chat_history,
        tools=[product_query_tool],
        tool_choice={"type": "function", "function": {"name": "should_query_product_database"}},
    )
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    return {
        "should_query": args.get("should_query", False),
        "reason": args.get("reason", ""),
        "follow_up": args.get("follow_up", "Could you tell me the model name or number?")
    }

def extract_model_name(chat_history: list[dict]) -> str:
    known_models = ["Astoria", "Astoria Pro", "Olivia", "PEAQ", "Air Handler Unit", "AHU", "Ceiling Cassette", "four-way ceiling cassette", "One Way Cassette", "High Static Slim Duct", "Medium Static Slim Duct", "Floor Ceiling Console", "MIA", "Outdoor Multi-Zone Hyper", "Outdoor Multi-Zone Regular"]
    combined_text = " ".join(m["content"] for m in chat_history if m["role"] == "user").lower()
    for model in known_models:
        if model.lower() in combined_text:
            return model
    return None

def query_snippets(user_input: str, model_name: str = None):
    embedding_response = openai_client.embeddings.create(
        input=user_input,
        model="text-embedding-3-small"
    )
    embedding = embedding_response.data[0].embedding
    filter_dict = {"model_name": {"$eq": model_name}} if model_name else {}
    response = index.query(
        vector=embedding,
        top_k=3,
        include_metadata=True,
        namespace=product_information,
        filter=filter_dict
    )
    return [
        f"[{m.metadata.get('title', 'Untitled')} - Page {m.metadata.get('page', 'N/A')}]: {m.metadata.get('content', '')}"
        for m in response.matches
    ]

def generate_response(user_input: str, context_snippets: list[str], chat_history: list[dict]) -> str:
    context_text = "\n\n".join(context_snippets)
    preamble = (
        "Use the following documentation snippets to answer the question clearly and accurately.\n\n"
        f"Documentation:\n{context_text}"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history
    messages.append({"role": "user", "content": preamble + f"\n\nUser Question: {user_input}\nAnswer:"})
    chat_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,
        max_tokens=300
    )
    return chat_response.choices[0].message.content.strip()

# --- Streamlit UI ---

st.set_page_config(page_title="Comfortside AI Agent", page_icon="🤖", layout="centered")
st.title("Comfortside AI Support.\nType your question below.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for i, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Ask me a question..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        try:
            decision = should_query_product_info(st.session_state.chat_history)
            if decision["should_query"]:
                model_name = extract_model_name(st.session_state.chat_history)
                snippets = query_snippets(user_input, model_name=model_name)
                if not snippets:
                    response = "Sorry, I couldn not find anything relevant."
                else:
                    response = generate_response(user_input, snippets, st.session_state.chat_history)
            else:
                response = decision.get("follow_up", decision.get("reason", "Let me know the model so I can help."))
        except Exception as e:
            response = f"❌ Error: {e}"

    # Append and show assistant message
    assistant_message = {"role": "assistant", "content": response}
    st.session_state.chat_history.append(assistant_message)

    with st.chat_message("assistant"):
        st.markdown(response)

    # ✅ Store latest exchange for feedback
    st.session_state.latest_user_input = user_input
    st.session_state.latest_response = response

# # Feedback logic (runs even after rerun)
# feedback_key_base = f"feedback_{len(st.session_state.chat_history)}"

# if f"{feedback_key_base}_submitted" not in st.session_state:
#     st.session_state[f"{feedback_key_base}_submitted"] = False

# if not st.session_state[f"{feedback_key_base}_submitted"] and "latest_user_input" in st.session_state and "latest_response" in st.session_state:
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("👍", key=f"{feedback_key_base}_up"):
#             st.session_state[f"{feedback_key_base}_submitted"] = True
#             st.success("Thanks for your feedback!")
#     with col2:
#         if st.button("👎", key=f"{feedback_key_base}_down"):
#             st.session_state[f"{feedback_key_base}_submitted"] = True
#             st.warning("Sorry about that — your feedback has been logged.")
#             try:
#                 feedback_entry = {
#                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#                     "user_question": st.session_state.latest_user_input,
#                     "ai_response": st.session_state.latest_response,
#                     "chat_history": st.session_state.chat_history.copy(),
#                     "query_snippets": snippets if "snippets" in locals() else []

#                 }
#                 with open("negative_feedback_log.jsonl", "a") as f:
#                     f.write(json.dumps(feedback_entry) + "\n")
#             except Exception as e:
#                 st.error(f"⚠️ Failed to save feedback: {e}")