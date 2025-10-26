# app.py (FINAL FIX - PERSISTENT EVENT LOOP FOR STREAMLIT)
import streamlit as st
import asyncio

# Import the core logic from your existing hybrid_chat.py file
from hybrid_chat import (
    pinecone_query,
    fetch_graph_context,
    summarize_pinecone_results,
    build_prompt,
    call_chat
)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Hybrid Travel Assistant",
    page_icon="✈️",
    layout="wide"
)

st.title("AI Hybrid Travel Assistant 🤖✈️")
# --- START OF FIX: BOLD THE DESCRIPTION ---
st.write("**Ask any question about planning a trip to Vietnam, and the assistant will generate an itinerary using a combination of vector search and knowledge graph data.**")
# --- END OF FIX ---

# --- START OF FINAL FIX ---
# 1. Manage the event loop manually and store it in Streamlit's session state.
# This ensures that the same loop is used across all reruns of the script.
if "event_loop" not in st.session_state:
    st.session_state.event_loop = asyncio.new_event_loop()

loop = st.session_state.event_loop
# --- END OF FINAL FIX ---

# --- Main App Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# This async function remains the same
# In app.py, replace the old get_ai_response function with this one:

async def get_ai_response(user_query: str):
    # --- START OF FIX: CONFIDENCE THRESHOLD ---
    CONFIDENCE_THRESHOLD = 0.5  # We'll only proceed if the top match has a score of 50% or higher
    # --- END OF FIX ---

    with st.spinner("Thinking... 🧠 (Fetching data from Pinecone and Neo4j...)"):
        pinecone_matches = await pinecone_query(user_query)

        # --- START OF FIX: CHECK FOR LOW CONFIDENCE ---
        # 1. Check if any matches were found at all
        if not pinecone_matches:
            return "I'm sorry, I couldn't find any relevant travel information for your query."

        # 2. Check if the score of the top match is below our threshold
        top_score = pinecone_matches[0].get('score', 0.0)
        if top_score < CONFIDENCE_THRESHOLD:
            return "I'm sorry, I couldn't find a confident match for your query. Could you please try rephrasing it?"
        # --- END OF FIX ---

        summary = summarize_pinecone_results(pinecone_matches)
        match_ids = [m["id"] for m in pinecone_matches]
        graph_facts = await fetch_graph_context(match_ids)
        final_prompt = build_prompt(user_query, pinecone_matches, graph_facts, summary)

    with st.spinner("Creative juices flowing... 🎨 (Generating itinerary with Gemini...)"):
        answer = await call_chat(final_prompt)
    
    return answer
if prompt := st.chat_input("e.g., create a romantic 4 day itinerary for Vietnam"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # --- START OF FINAL FIX ---
        # 2. Use the persistent loop to run the async function.
        # This replaces asyncio.run() and avoids creating/closing new loops.
        response = loop.run_until_complete(get_ai_response(prompt))
        # --- END OF FINAL FIX ---
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})