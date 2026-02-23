#  Graph RAG Chatbot

A chatbot that uses a **Knowledge Graph as memory**. It extracts facts from natural language, stores them as graph nodes/edges, and retrieves relevant context when you ask questions.

---

##  Quick Setup 

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Get a free Groq API key
→ Go to [console.groq.com](https://console.groq.com) → Sign up → Create API key (it's free)

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Enter your Groq API key in the sidebar, set a username, and start chatting!

---

##  Example Inputs (10 facts to add)

Paste these into the chat one by one to populate the graph:

1. `Ahan is a software engineer who works at Google in San Francisco.`
2. `Bob is Ahan's manager and he has been at Google for 5 years.`
3. `Ahan is learning machine learning and enjoys hiking on weekends.`
4. `The project Orion is led by Ahan and uses Python and TensorFlow.`
5. `Bob previously worked at Meta before joining Google.`
6. `Ahan has a dog named Max who loves the beach.`
7. `The team uses Slack and Jira for project management.`
8. `Ahan graduated from IIT Delhi with a degree in Computer Science.`
9. `Bob is married to Carol who is a doctor at Stanford Hospital.`
10. `The Orion project deadline is March 2026 and has a budget of $500k.`

---

##  Example Queries (5 questions using graph context)

After adding facts above, try:

1. `Who is Ahan?` → Should return: engineer at Google, IIT Delhi grad, leads Orion project
2. `What project does Ahan lead?` → Orion project, Python/TensorFlow, March 2026 deadline
3. `Tell me about Bob.` → Ahan's manager, 5 years at Google, previously at Meta, married to Carol
4. `What tools does the team use?` → Slack, Jira
5. `Where did Ahan study?` → IIT Delhi, Computer Science degree

---

##  Architecture

```
User Input
    │
    ├─► Is it a STATEMENT? ──► LLM + spaCy extract triplets (subject, relation, object)
    │                               │
    │                               ▼
    │                         NetworkX DiGraph ──► Saved per user as .pkl
    │
    └─► Is it a QUESTION? ──► Query embedding/token match against graph nodes
                                    │
                                    ▼
                              Retrieve top-k triplets (subgraph context)
                                    │
                                    ▼
                              LLM answers using context (Groq llama3-8b)
```

---

##  Graph Visualisation

The live graph renders in the right panel using **pyvis** (built on vis.js):
- **Node size** = number of mentions
- **Node color** = degree (blue=low, red=high connectivity)
- **Edge labels** = relationship type
- Fully interactive: drag, zoom, hover for details

---

##  File Structure

```
graphrag/
├── app.py              # Main Streamlit app
├── requirements.txt    # Dependencies
├── graphs/             # Per-user graph storage (auto-created)
│   ├── user1_graph.pkl
│   └── user2_graph.pkl
└── README.md
```

---

##  Reflection 

### What the extraction pipeline handles well and where it breaks

**Handles well:**
- Clear subject-verb-object sentences: *"Ahan works at Google"* → `(ahan, works_at, google)` ✅
- Named entities (people, orgs, locations) via spaCy NER — these become reliable nodes
- LLM extraction handles complex sentences like *"The project led by Ahan uses Python"*
- Compound facts in single sentences get decomposed into multiple triplets
- Relationship types are preserved as edge labels, not lost

**Where it breaks:**
- **Pronouns**: *"He went there yesterday"* — spaCy can't resolve "he" or "there" without context window
- **Implicit facts**: *"Ahan is brilliant"* is an attribute, not a clear triplet; adjective-based facts often get dropped
- **Numerical data**: *"Budget of $500k"* — numbers often don't anchor well as graph nodes
- **Negations**: *"Ahan does NOT work at Meta"* may still create an `ahan → works_at → meta` edge
- **Context-dependent statements**: Sarcasm, metaphors, conditionals all confuse extraction

### One limitation of the graph structure for memory

The biggest limitation is **graph sparsity vs. semantic richness**. A knowledge graph stores *explicit* relationships as discrete edges, but human memory is associative and contextual. If you say *"Ahan loves Italian food and so does her dog Max"*, the graph creates a node for "Italian food" connected to both Ahan and Max — but it loses the nuance that this is a *shared preference* between owner and pet, not a causal link. Graph traversal is rigid: you either have a path between two nodes or you don't. Unlike vector databases or full conversation history, graphs can't infer *implicit* connections or handle analogical reasoning ("Ahan is like Bob in that..."). This means the retrieval step may miss relevant context if the query uses synonyms or paraphrases that don't match node labels exactly.

---

##  Notes

- **Multi-user**: Each username gets its own isolated graph file under `graphs/`
- **No API key mode**: Use spaCy-only extraction (less accurate but works offline)
- Groq's `llama-3.1-8b-instant` is **free** and very fast (~1-2s responses)
