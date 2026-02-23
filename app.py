import streamlit as st
import networkx as nx
import spacy
import json
import os
import re
from groq import Groq
from pyvis.network import Network
import streamlit.components.v1 as components
import pickle
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GRAPH_FILE = "graphs/{user}_graph.pkl"
os.makedirs("graphs", exist_ok=True)

st.set_page_config(page_title="Graph RAG Chatbot", layout="wide")

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_nlp()

# ─── GRAPH UTILITIES ──────────────────────────────────────────────────────────
def load_graph(user: str) -> nx.DiGraph:
    path = GRAPH_FILE.format(user=user)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return nx.DiGraph()

def save_graph(G: nx.DiGraph, user: str):
    path = GRAPH_FILE.format(user=user)
    with open(path, "wb") as f:
        pickle.dump(G, f)

def extract_triplets_spacy(text: str):
    """Extract (subject, relation, object) triplets using spaCy dependency parse."""
    doc = nlp(text)
    triplets = []
    
    for token in doc:
        if token.dep_ in ("ROOT", "relcl") and token.pos_ == "VERB":
            subj = None
            obj = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj = " ".join([t.text for t in child.subtree 
                                     if t.dep_ not in ("punct",)])
                if child.dep_ in ("dobj", "attr", "pobj", "prep"):
                    obj = " ".join([t.text for t in child.subtree 
                                    if t.dep_ not in ("punct",)])
            if subj and obj:
                triplets.append((subj.strip(), token.lemma_, obj.strip()))
    
    # Also extract named entity pairs that co-occur
    ents = [(e.text, e.label_) for e in doc.ents]
    for i in range(len(ents)):
        for j in range(i+1, len(ents)):
            e1, l1 = ents[i]
            e2, l2 = ents[j]
            triplets.append((e1, f"related_to", e2))
    
    return triplets

def llm_extract_triplets(text: str, client: Groq) -> list:
    """Use LLM to extract triplets from text."""
    prompt = f"""Extract factual triplets from this text as JSON array.
Each triplet: {{"subject": "...", "relation": "...", "object": "..."}}
Keep subjects/objects as concise noun phrases. Relations as verbs/phrases.

Text: {text}

Return ONLY a JSON array, nothing else. Example:
[{{"subject": "Alice", "relation": "works at", "object": "Google"}}, {{"subject": "Alice", "relation": "lives in", "object": "New York"}}]"""
    
    try:
        resp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512
        )
        raw = resp.choices[0].message.content.strip()
        # Extract JSON array
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            triplets_data = json.loads(match.group())
            return [(t["subject"], t["relation"], t["object"]) for t in triplets_data]
    except Exception as e:
        st.warning(f"LLM extraction fallback to spaCy: {e}")
    return []

def add_to_graph(G: nx.DiGraph, triplets: list, source_text: str, user: str):
    """Add triplets to the knowledge graph."""
    added = []
    for subj, rel, obj in triplets:
        subj_clean = subj.lower().strip()
        obj_clean = obj.lower().strip()
        if len(subj_clean) < 2 or len(obj_clean) < 2:
            continue
        
        # Add nodes with metadata
        if not G.has_node(subj_clean):
            G.add_node(subj_clean, label=subj, mentions=1, 
                       first_seen=datetime.now().isoformat())
        else:
            G.nodes[subj_clean]['mentions'] = G.nodes[subj_clean].get('mentions', 0) + 1
            
        if not G.has_node(obj_clean):
            G.add_node(obj_clean, label=obj, mentions=1,
                       first_seen=datetime.now().isoformat())
        
        # Add edge with relation
        if G.has_edge(subj_clean, obj_clean):
            # Append relation if different
            existing = G[subj_clean][obj_clean].get('relation', '')
            if rel not in existing:
                G[subj_clean][obj_clean]['relation'] = f"{existing}, {rel}"
        else:
            G.add_edge(subj_clean, obj_clean, relation=rel, source=source_text[:100])
            added.append((subj_clean, rel, obj_clean))
    
    return added

def retrieve_context(G: nx.DiGraph, query: str, top_k: int = 10) -> str:
    """Retrieve relevant graph context for a query."""
    if len(G.nodes) == 0:
        return "No information stored yet."
    
    query_doc = nlp(query.lower())
    query_tokens = set([t.lemma_ for t in query_doc if not t.is_stop and len(t.text) > 2])
    
    # Score nodes by relevance to query
    node_scores = {}
    for node in G.nodes:
        score = 0
        node_tokens = set(node.lower().split())
        # Token overlap
        score += len(query_tokens & node_tokens) * 2
        # Substring match
        for qt in query_tokens:
            if qt in node:
                score += 1
        node_scores[node] = score
    
    # Get top nodes
    top_nodes = sorted(node_scores, key=lambda x: node_scores[x], reverse=True)[:5]
    top_nodes = [n for n in top_nodes if node_scores[n] > 0]
    
    if not top_nodes:
        # Fall back to all edges if no match
        top_nodes = list(G.nodes)[:5]
    
    # Gather triplets from top nodes and their neighbors
    context_triplets = []
    visited = set()
    for node in top_nodes:
        # Outgoing edges
        for _, nbr, data in G.out_edges(node, data=True):
            triplet = f"{G.nodes[node].get('label', node)} --[{data.get('relation','')}]--> {G.nodes[nbr].get('label', nbr)}"
            if triplet not in visited:
                context_triplets.append(triplet)
                visited.add(triplet)
        # Incoming edges
        for src, _, data in G.in_edges(node, data=True):
            triplet = f"{G.nodes[src].get('label', src)} --[{data.get('relation','')}]--> {G.nodes[node].get('label', node)}"
            if triplet not in visited:
                context_triplets.append(triplet)
                visited.add(triplet)
    
    return "\n".join(context_triplets[:top_k]) if context_triplets else "No relevant context found."

def chat_with_context(query: str, context: str, history: list, client: Groq) -> str:
    """Generate answer using retrieved graph context."""
    system = """You are a helpful assistant with access to a personal knowledge graph.
Use the provided graph context (knowledge triplets) to answer questions accurately.
If the context doesn't contain enough info, say so honestly.
Be concise and direct."""
    
    context_msg = f"Knowledge Graph Context:\n{context}\n\nQuestion: {query}"
    
    messages = [{"role": "system", "content": system}]
    # Add last 4 turns of history
    for h in history[-4:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": context_msg})
    
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.3,
        max_tokens=512
    )
    return resp.choices[0].message.content

def render_graph(G: nx.DiGraph) -> str:
    """Render graph as interactive HTML using pyvis."""
    net = Network(height="500px", width="100%", bgcolor="#0f1117", 
                  font_color="white", directed=True)
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)
    
    # Color nodes by degree
    max_deg = max((G.degree(n) for n in G.nodes), default=1)
    
    for node in G.nodes:
        deg = G.degree(node)
        # Color gradient: low degree = blue, high = red
        intensity = int(255 * deg / max(max_deg, 1))
        color = f"#{intensity:02x}{100:02x}{255-intensity:02x}"
        label = G.nodes[node].get('label', node)
        mentions = G.nodes[node].get('mentions', 1)
        size = 15 + min(mentions * 3, 30)
        net.add_node(node, label=label, color=color, size=size,
                     title=f"{label}\nDegree: {deg}\nMentions: {mentions}")
    
    for src, dst, data in G.edges(data=True):
        rel = data.get('relation', '')
        net.add_edge(src, dst, label=rel, title=rel, 
                     color="#666688", arrows="to")
    
    # Return HTML
    net.set_options("""
    var options = {
      "edges": {"font": {"size": 10, "color": "#aaaacc"}},
      "physics": {"enabled": true, "stabilization": {"iterations": 100}}
    }
    """)
    html = net.generate_html()
    return html

# ─── STREAMLIT UI ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
* { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f1117; color: #e0e0f0; }
.stTextInput input { background: #1a1d2e; border: 1px solid #333355; color: white; border-radius: 8px; }
.stButton button { background: linear-gradient(135deg, #6e40c9, #3b82f6); color: white; border: none; border-radius: 8px; font-weight: 600; }
.chat-msg-user { background: #1e2139; border-left: 3px solid #6e40c9; padding: 10px 14px; border-radius: 8px; margin: 6px 0; }
.chat-msg-bot { background: #151829; border-left: 3px solid #3b82f6; padding: 10px 14px; border-radius: 8px; margin: 6px 0; }
.triplet-pill { display: inline-block; background: #1e2139; border: 1px solid #3b82f6; border-radius: 20px; padding: 2px 10px; margin: 2px; font-size: 12px; color: #a0c0ff; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }
</style>
""", unsafe_allow_html=True)

st.title("Graph RAG Chatbot")
st.caption("Tell me things. Ask me things. I remember with a knowledge graph.")

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    
    api_key = st.text_input("Groq API Key", type="password", 
                             help="Get free key at console.groq.com",
                             value=os.environ.get("GROQ_API_KEY", ""))
    
    st.divider()
    username = st.text_input("👤 Username", value="user1", 
                              help="Each user has their own graph memory")
    
    st.divider()
    extraction_method = st.radio("Extraction Method", 
                                  ["LLM (better)", "spaCy (faster, no API)"])
    
    st.divider()
    if st.button("🗑️ Clear My Graph"):
        path = GRAPH_FILE.format(user=username)
        if os.path.exists(path):
            os.remove(path)
        st.session_state.pop('graph_cache', None)
        st.success("Graph cleared!")
        st.rerun()
    
    st.divider()
    st.markdown("**📊 Graph Stats**")
    G_temp = load_graph(username)
    st.metric("Nodes", len(G_temp.nodes))
    st.metric("Edges", len(G_temp.edges))

# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "graph_version" not in st.session_state:
    st.session_state.graph_version = 0

# ─── MAIN LAYOUT ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Chat")
    
    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-msg-user">👤 {msg["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-msg-bot">🤖 {msg["content"]}</div>',
                           unsafe_allow_html=True)
                if msg.get("triplets"):
                    st.markdown("**Extracted facts:**")
                    pills = " ".join([f'<span class="triplet-pill">{s} → {r} → {o}</span>' 
                                     for s,r,o in msg["triplets"][:6]])
                    st.markdown(pills, unsafe_allow_html=True)
                if msg.get("context_used"):
                    with st.expander("📎 Graph context used"):
                        st.code(msg["context_used"], language=None)
    
    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Message", placeholder="Tell me about yourself, or ask a question...", label_visibility="collapsed")
        submitted = st.form_submit_button("Send →", use_container_width=True)
    
    # Example inputs
    with st.expander(" Try these example inputs"):
        examples = [
            "Ahan is a software engineer who works at Google in San Francisco.",
            "Bob is Ahan's manager and he has been at Google for 5 years.",
            "Ahan is learning machine learning and enjoys hiking on weekends.",
            "The project Orion is led by Ahan and uses Python and TensorFlow.",
            "Bob previously worked at Meta before joining Google.",
            "Ahan has a dog named Max who loves the beach.",
            "The team uses Slack and Jira for project management.",
            "Ahan graduated from IIT Delhi with a degree in Computer Science.",
            "Bob is married to Carol who is a doctor at Stanford Hospital.",
            "The Orion project deadline is March 2026 and has a budget of $500k.",
        ]
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state['prefill'] = ex
                st.rerun()

with col2:
    st.subheader("Knowledge Graph")
    G = load_graph(username)
    
    if len(G.nodes) > 0:
        graph_html = render_graph(G)
        components.html(graph_html, height=520, scrolling=False)
    else:
        st.info("Your knowledge graph is empty. Start telling me things!")
        st.markdown("""
        **How it works:**
        1. Tell the bot facts (people, places, relationships)
        2. Facts get extracted and stored as graph nodes/edges  
        3. When you ask questions, relevant graph context is retrieved
        4. The LLM answers using that context
        """)

# ─── HANDLE SUBMIT ────────────────────────────────────────────────────────────
# Handle prefill from example buttons
if 'prefill' in st.session_state:
    submitted = True
    user_input = st.session_state.pop('prefill')

if submitted and user_input and user_input.strip():
    if not api_key and "LLM" in extraction_method:
        st.error("Please enter your Groq API key in the sidebar. Get one free at console.groq.com")
    else:
        G = load_graph(username)
        
        # Determine if this is a question or a statement
        is_question = "?" in user_input or user_input.lower().startswith(
            ("what", "who", "where", "when", "how", "why", "is", "are", "does", "did", "can", "tell me"))
        
        response_data = {"role": "assistant", "content": "", "triplets": [], "context_used": ""}
        
        with st.spinner("Processing..."):
            if is_question:
                # RETRIEVAL MODE
                context = retrieve_context(G, user_input)
                response_data["context_used"] = context
                
                if api_key:
                    client = Groq(api_key=api_key)
                    answer = chat_with_context(user_input, context, 
                                               st.session_state.chat_history, client)
                else:
                    answer = f"Based on stored knowledge:\n{context}"
                
                response_data["content"] = answer
            else:
                # STORAGE MODE - extract and store facts
                triplets = []
                
                if api_key and "LLM" in extraction_method:
                    client = Groq(api_key=api_key)
                    triplets = llm_extract_triplets(user_input, client)
                
                # Always augment with spaCy
                spacy_triplets = extract_triplets_spacy(user_input)
                # Combine, deduplicate
                all_triplets = triplets + [t for t in spacy_triplets if t not in triplets]
                
                added = add_to_graph(G, all_triplets, user_input, username)
                save_graph(G, username)
                
                response_data["triplets"] = added
                
                if added:
                    response_data["content"] = f"✅ Got it! I stored {len(added)} new fact(s) in your knowledge graph."
                else:
                    response_data["content"] = "I processed that but couldn't extract clear facts. Try stating things more directly like 'Alice works at Google'."
                
                st.session_state.graph_version += 1
        
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append(response_data)
        st.rerun()