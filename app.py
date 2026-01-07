import streamlit as st
import sqlite3
import os
import chromadb
import uuid
from datetime import datetime
from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# 1. CONFIGURATION & INITIALIZATION
# ---------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="The Stratagem Journal", layout="wide")

# Initialize Local Database (SQLite) for Structured Data
def init_db():
    conn = sqlite3.connect('stratagem.db')
    c = conn.cursor()
    # Journal Entries Table
    c.execute('''CREATE TABLE IF NOT EXISTS journal
                 (id TEXT PRIMARY KEY, date TEXT, category TEXT, content TEXT, analysis TEXT)''')
    # Protocols Table (Active Habits/Strategies)
    c.execute('''CREATE TABLE IF NOT EXISTS protocols
                 (id TEXT PRIMARY KEY, name TEXT, source TEXT, status TEXT, start_date TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Initialize Vector DB (Chroma) for Memory & Library
PERSIST_DIRECTORY = "./chroma_db"
embeddings = OpenAIEmbeddings()
# We use two collections: one for Journal History, one for the Library (Books)
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

# Initialize AI Model
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# ---------------------------------------------------------
# 2. AGENT PERSONAS (The "How to Win" Framework)
# ---------------------------------------------------------

# Agent A: The Red Teamer (Career/Strategy)
# Philosophy: Rule 1 & 2 - Comfort doesn't sharpen you; complexity attracts layered strategy.
red_team_prompt = ChatPromptTemplate.from_template("""
You are 'The Red Teamer', a strategic opponent designed to sharpen the user.
Your Philosophy:
1. "Comfort doesn’t sharpen you; a superior opponent exposes blind spots." [cite: 7, 8]
2. "Treat every loss as data" and "emulate process, not outcomes." [cite: 9, 10]
3. If the situation is complex, look for "layered strategy" and "multi-step gambits." 

Task: Analyze the user's journal entry.
- Identify where they accepted 'answers' instead of asking 'questions'.
- Map stakeholder incentives they might have missed.
- Critique their specific decisions (not their results).

User Entry: {entry}
Analysis:
""")

# Agent B: The Ego Surgeon (Personal/Emotion)
# Philosophy: The Surgical Method - Name the voice, ask adversarial questions.
ego_surgeon_prompt = ChatPromptTemplate.from_template("""
You are 'The Ego Surgeon'. Your goal is to identify the 'Ego-Opponent' hiding in the user's narrative.
Your Philosophy:
1. The ego disguises itself as the user's best friend but uses 'rationalizations' to trap them. [cite: 24]
2. Use the "Surgical Method":
   - "Name the voice" (e.g., The Voice of Shame, The Defensive Voice). [cite: 35, 39]
   - "Adversarial Questioning": Ask "Who benefits if this thought is true?" [cite: 36, 42]
   - "Evidence Audit": List objective facts vs. feelings. [cite: 42]

Task: Analyze the user's journal entry for emotional fusion.
User Entry: {entry}
Diagnosis & Surgical Steps:
""")

# Agent C: The Architect (RAG/Library)
# Philosophy: Operationalizing external knowledge.
architect_prompt = ChatPromptTemplate.from_template("""
You are 'The Architect'. You bridge external knowledge (Books) with internal reality (User Journal).
Context from Library (Books): {book_context}
User's Current Struggle: {query}

Task:
1. Extract a specific 'Protocol' or 'Mental Model' from the book context.
2. Apply it directly to the user's struggle.
3. Define a measurable test: "How will we know this worked in 7 days?"
""")

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------

def save_entry(category, content, analysis):
    entry_id = str(uuid.uuid4())
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Save to SQLite
    conn = sqlite3.connect('stratagem.db')
    c = conn.cursor()
    c.execute("INSERT INTO journal VALUES (?, ?, ?, ?, ?)", 
              (entry_id, date_str, category, content, analysis))
    conn.commit()
    conn.close()
    
    # 2. Save to Vector DB (for future retrieval)
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
    # Save temp file
    with open(f"temp_{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load & Split
    loader = PyPDFLoader(f"temp_{uploaded_file.name}")
    pages = loader.load_and_split()
    
    # Add to Library Vector Store
    library_collection.add_documents(pages)
    os.remove(f"temp_{uploaded_file.name}")
    return len(pages)

# ---------------------------------------------------------
# 4. STREAMLIT UI LAYOUT
# ---------------------------------------------------------

st.title("♟️ The Stratagem Journal")
st.markdown("*Operationalizing 'How to Win at Any Game or Con'*")

tabs = st.tabs(["📝 Daily Journal", "📚 The Library (RAG)", "⚙️ Protocols", "🗄️ Archives"])

# --- TAB 1: DAILY JOURNAL ---
with tabs[0]:
    st.header("The Raw Stream")
    category = st.selectbox("Context", ["Career / Strategy", "Personal / Ego"])
    journal_text = st.text_area("Log your raw experience, decisions, and emotions...", height=200)
    
    if st.button("Submit & Analyze"):
        with st.spinner("The Agents are Red Teaming your entry..."):
            # 1. Run Analysis
            analysis_result = process_journal(category, journal_text)
            
            # 2. Display Results
            st.subheader("🕵️ Agent Analysis")
            st.markdown(analysis_result)
            
            # 3. Save Data
            save_entry(category, journal_text, analysis_result)
            st.success("Entry encrypted and archived in Portfolio.")

# --- TAB 2: THE LIBRARY (RAG) ---
with tabs[1]:
    st.header("The Knowledge Lab")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Strategy Docs (PDF)", type="pdf")
    if uploaded_file:
        if st.button("Ingest to Library"):
            with st.spinner("Parsing and embedding knowledge..."):
                num_pages = ingest_file(uploaded_file)
                st.success(f"Ingested {num_pages} pages into the Vector Database.")

    st.divider()
    
    # Brainstorming Interface
    st.subheader("Brainstorm with The Architect")
    user_query = st.text_input("Ask a question based on your Library & Past Journal:")
    
    if user_query and st.button("Consult Architect"):
        # Retrieve relevant book chunks
        results = library_collection.similarity_search(user_query, k=3)
        context_text = "\n\n".join([doc.page_content for doc in results])
        
        # Run Architect Chain
        chain = architect_prompt | llm | StrOutputParser()
        advice = chain.invoke({"book_context": context_text, "query": user_query})
        
        st.info("Based on your library:")
        st.markdown(advice)

# --- TAB 3: PROTOCOLS ---
with tabs[2]:
    st.header("Active Protocols")
    
    # Add New Protocol
    with st.expander("Install New Protocol"):
        new_proto_name = st.text_input("Protocol Name (e.g., '2-Minute Rule')")
        new_proto_source = st.text_input("Source (e.g., 'Atomic Habits')")
        if st.button("Activate Protocol"):
            conn = sqlite3.connect('stratagem.db')
            c = conn.cursor()
            c.execute("INSERT INTO protocols VALUES (?, ?, ?, ?, ?)", 
                      (str(uuid.uuid4()), new_proto_name, new_proto_source, "Active", datetime.now().strftime("%Y-%m-%d")))
            conn.commit()
            conn.close()
            st.rerun()

    # View Active Protocols
    conn = sqlite3.connect('stratagem.db')
    df_protocols = c.execute("SELECT name, source, start_date FROM protocols WHERE status='Active'").fetchall()
    conn.close()
    
    if df_protocols:
        for p in df_protocols:
            st.metric(label=f"{p[0]} ({p[1]})", value=f"Started: {p[2]}")
    else:
        st.write("No active protocols. Go to the Library to extract one.")

# --- TAB 4: ARCHIVES ---
with tabs[3]:
    st.header("Journal History")
    conn = sqlite3.connect('stratagem.db')
    c = conn.cursor()
    entries = c.execute("SELECT date, category, content, analysis FROM journal ORDER BY date DESC").fetchall()
    conn.close()
    
    for date, cat, content, analysis in entries:
        with st.expander(f"{date} - {cat}"):
            st.markdown(f"**Raw Entry:** {content}")
            st.divider()
            st.markdown(f"**Analysis:**\n{analysis}")
