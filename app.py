__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import sqlite3
import os
import chromadb
import uuid
from datetime import datetime
from dotenv import load_dotenv

# --- GOOGLE GEMINI IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# 1. CONFIGURATION & INITIALIZATION
# ---------------------------------------------------------
st.set_page_config(page_title="The Stratagem Journal", layout="wide")

# Secure API Key Retrieval for Streamlit Cloud
# On local machine, it looks for .env. On Cloud, it looks for st.secrets
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        st.error("⚠️ GOOGLE_API_KEY not found. Please set it in Streamlit Secrets.")
        st.stop()

# Initialize Local Database (SQLite)
# NOTE: On Streamlit Free Tier, this file resets on reboot. 
# For permanent storage, connect to Supabase or Google Sheets later.
def init_db():
    conn = sqlite3.connect('stratagem.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS journal
                 (id TEXT PRIMARY KEY, date TEXT, category TEXT, content TEXT, analysis TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS protocols
                 (id TEXT PRIMARY KEY, name TEXT, source TEXT, status TEXT, start_date TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Initialize Vector DB (Chroma)
# We use the specific Google Embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Persistent Client setup
PERSIST_DIRECTORY = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

journal_collection = Chroma(
    client=chroma_client,
    collection_name="journal_memory",
    embedding_function=embeddings,
)
library_collection = Chroma(
    client=chroma_client,
    collection_name="knowledge_library",
    embedding_function=embeddings,
)

# Initialize Gemini Model
# gemini-1.5-flash is fast and free-tier friendly
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=api_key,
    convert_system_message_to_human=True # Fix for some Gemini system prompt nuances
)

# ---------------------------------------------------------
# 2. AGENT PERSONAS ("How to Win" Framework)
# ---------------------------------------------------------

# Agent A: The Red Teamer
red_team_prompt = ChatPromptTemplate.from_template("""
You are 'The Red Teamer'.
Philosophy:
1. [cite_start]"Comfort doesn’t sharpen you; a superior opponent exposes blind spots." [cite: 7, 8]
2. [cite_start]"The More Sophisticated the Game, the More Sophisticated the Opponent." [cite: 10]
3. [cite_start]"The Game Stops When You Start Giving Answers." [cite: 14]

Task: Analyze the user's journal entry.
- Identify where they accepted 'answers' instead of asking 'questions'.
- Map stakeholder incentives they might have missed.
- Critique their specific decisions (not their results).

User Entry: {entry}
Analysis:
""")

# Agent B: The Ego Surgeon
ego_surgeon_prompt = ChatPromptTemplate.from_template("""
You are 'The Ego Surgeon'.
Philosophy:
1. [cite_start]The ego is the opponent. It "layers sophistication" and "provides ready-made answers." [cite: 24, 29]
2. Use the "Surgical Method":
   - [cite_start]"Name the voice" (e.g., The Voice of Shame). [cite: 35]
   - [cite_start]"Adversarial Questioning": Ask "Who benefits if this thought is true?" [cite: 36]
   - [cite_start]"Evidence Audit": List objective facts vs. feelings. [cite: 42]

Task: Analyze the user's journal entry for emotional fusion.
User Entry: {entry}
Diagnosis & Surgical Steps:
""")

# Agent C: The Architect
architect_prompt = ChatPromptTemplate.from_template("""
You are 'The Architect'. You bridge external knowledge (Books) with internal reality.
Context from Library: {book_context}
User's Current Struggle: {query}

Task:
1. Extract a specific 'Protocol' from the book context.
2. Apply it directly to the user's struggle.
3. Define a measurable test for the next 7 days.
""")

# ---------------------------------------------------------
# 3. LOGIC FUNCTIONS
# ---------------------------------------------------------

def save_entry(category, content, analysis):
    entry_id = str(uuid.uuid4())
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # SQLite Save
    conn = sqlite3.connect('stratagem.db')
    c = conn.cursor()
    c.execute("INSERT INTO journal VALUES (?, ?, ?, ?, ?)", 
              (entry_id, date_str, category, content, analysis))
    conn.commit()
    conn.close()
    
    # Vector Save
    journal_collection.add_documents(documents=[
        RecursiveCharacterTextSplitter().create_documents([content], metadatas=[{"date": date_str, "category": category}])[0]
    ])

def process_journal(category, content):
    if category == "Career / Strategy":
        chain = red_team_prompt | llm | StrOutputParser()
        return chain.invoke({"entry": content})
    else:
        chain = ego_surgeon_prompt | llm | StrOutputParser()
        return chain.invoke({"entry": content})

def ingest_file(uploaded_file):
    with open(f"temp_{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader(f"temp_{uploaded_file.name}")
    pages = loader.load_and_split()
    
    library_collection.add_documents(pages)
    os.remove(f"temp_{uploaded_file.name}")
    return len(pages)

# ---------------------------------------------------------
# 4. UI LAYOUT
# ---------------------------------------------------------

st.title("♟️ The Stratagem Journal (Gemini Core)")
st.markdown("*Operationalizing 'How to Win at Any Game or Con'*")

tabs = st.tabs(["📝 Daily Journal", "📚 The Library", "⚙️ Protocols", "🗄️ Archives"])

# --- TAB 1: JOURNAL ---
with tabs[0]:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("The Raw Stream")
        category = st.selectbox("Context", ["Career / Strategy", "Personal / Ego"])
        journal_text = st.text_area("Log your raw experience...", height=200)
        
        if st.button("Submit & Analyze"):
            if not journal_text:
                st.warning("Please write something first.")
            else:
                with st.spinner("Gemini Agents are Red Teaming your entry..."):
                    analysis_result = process_journal(category, journal_text)
                    st.success("Analysis Complete")
                    st.markdown("### 🕵️ Agent Report")
                    st.markdown(analysis_result)
                    save_entry(category, journal_text, analysis_result)

    with col2:
        st.info("💡 **Tip:** Be honest. The Ego Surgeon is watching for 'ready-made answers'.")

# --- TAB 2: LIBRARY ---
with tabs[1]:
    st.header("The Knowledge Lab")
    uploaded_file = st.file_uploader("Upload Strategy Docs (PDF)", type="pdf")
    if uploaded_file and st.button("Ingest"):
        with st.spinner("Embedding knowledge..."):
            num = ingest_file(uploaded_file)
            st.success(f"Ingested {num} pages.")

    st.divider()
    user_query = st.text_input("Brainstorm with The Architect:")
    if user_query and st.button("Consult"):
        results = library_collection.similarity_search(user_query, k=3)
        context = "\n\n".join([doc.page_content for doc in results])
        chain = architect_prompt | llm | StrOutputParser()
        res = chain.invoke({"book_context": context, "query": user_query})
        st.markdown(res)

# --- TAB 3: PROTOCOLS ---
with tabs[2]:
    st.header("Active Protocols")
    with st.expander("Add New Protocol"):
        p_name = st.text_input("Protocol Name")
        p_src = st.text_input("Source")
        if st.button("Activate"):
            conn = sqlite3.connect('stratagem.db')
            c = conn.cursor()
            c.execute("INSERT INTO protocols VALUES (?, ?, ?, ?, ?)", 
                      (str(uuid.uuid4()), p_name, p_src, "Active", datetime.now().strftime("%Y-%m-%d")))
            conn.commit()
            conn.close()
            st.rerun()

    conn = sqlite3.connect('stratagem.db')
    data = conn.execute("SELECT name, source, start_date FROM protocols").fetchall()
    conn.close()
    if data:
        for d in data:
            st.success(f"**{d[0]}** (Source: {d[1]}) - Started: {d[2]}")
    else:
        st.write("No active protocols.")

# --- TAB 4: ARCHIVES ---
with tabs[3]:
    st.header("History")
    if st.button("Refresh Archives"):
        st.rerun()
        
    conn = sqlite3.connect('stratagem.db')
    rows = conn.execute("SELECT date, category, content, analysis FROM journal ORDER BY date DESC").fetchall()
    conn.close()
    
    for r in rows:
        with st.expander(f"{r[0]} | {r[1]}"):
            st.write(r[2])
            st.divider()
            st.caption("Analysis")
            st.write(r[3])
