# 🎥 Production YouTube RAG Engine

Upload a YouTube video, ask questions, and receive answers grounded entirely in the transcript. Built using LangChain, OpenAI, and FAISS to minimize hallucinations and support structured outputs.

🚀 **Live Demo:** https://youtube-rag-engine.streamlit.app/

---

## 📖 How It Works (For Everyone)

This system is built to act like a super-intelligent research assistant for video content. Instead of making you watch a 2-hour podcast or video, it reads the transcript, organizes the information, and answers grounded in retrieved transcript content.

### 🧠 The Non-Technical View: The 3-Step Lifecycle
* **Step 1: Ingestion (The Reader):** The app takes a YouTube link, contacts YouTube's database, and grabs the full text of everything spoken in the video.
* **Step 2: Processing & Storage (The Filing Cabinet):** Instead of cutting the text mid-sentence, it reads the *meaning* of the words and cuts the text into neatly organized paragraphs whenever the topic changes. It then stores these in a highly organized digital filing cabinet.
* **Step 3: Retrieval & Guardrails (The Secure Assistant):** When you ask a question, the assistant runs to the filing cabinet, grabs the exact paragraphs that mention your topic, and reads them. **Here is the catch:** If you ask a question about something *not* mentioned in the video, the assistant is strictly trained to say, *"The video context does not contain that information"* instead of guessing or making things up.

### 💻 The Technical View: Advanced AI Engineering
* **Semantic Chunking (`langchain_experimental`):** Bypasses traditional token-count or character-boundary splitting. It maps text sentences into an embedding space and computes the statistical distance between consecutive sentences. A split is dynamically created *only* when the semantic distance exceeds a threshold, preserving the full context of long-form stories.
* **Vector Space & FAISS Caching:** Generates dense vector representations using OpenAI's `text-embedding-3-small` model. These are indexed and stored locally via a high-performance binary **FAISS (Facebook AI Similarity Search)** vector database to allow for lightning-fast, localized semantic retrieval without system overhead.
* **Maximal Marginal Relevance (MMR) Search:** To avoid feeding the LLM redundant context, the retrieval layer implements MMR. This optimizes for both **similarity to the query** and **diversity among the results**, ensuring the LLM gets multiple angles of information instead of the same sentence phrased three different ways.
* **Deterministic Structured Outputs via Pydantic:** Integrates rigid Pydantic data schemas directly into the `gpt-4o-mini` generation call. This guarantees that the final executive output always complies with a strict JSON layout, making it completely reliable for downstream applications.

---

## 🛠️ System Architecture Diagram

```text
[YouTube Link] ──> [YouTube Transcript API] ──> [Semantic Text Splitter]
                                                        │
                                                        ▼
[FAISS Vector Store] <── [text-embedding-3-small] <── [Text Chunks]
        │
        ├── (MMR Search Filter) ──> [Diverse Context Blocks]
                                              │
                                              ▼
[User Query] ───────────────────────────> [gpt-4o-mini + Pydantic Schema]
                                              │
                                              ▼
                              [Deterministic JSON Output]
                              (Or Strict Factual Refusal)

📦 Project Structure
YOUTUBE_CHATBOT/
├── faiss_index/          # Localized cached binary vector database files
│   ├── index.faiss       # The raw semantic vector index
│   └── index.pkl         # Serialized chunk metadata mappings
├── app.py                # Streamlit UI Layer & multi-mode dashboard orchestration
├── ingestion.py          # Data ingestion, Semantic Chunking, and Vector Indexing
├── main.py               # Central execution pipeline controller
├── retriever.py          # MMR retrieval layer and LLM schema validation
├── requirements.txt      # Production package dependencies
└── README.md             # Systems documentation

🚀 Getting Started1.
Installation
Clone the repository and install the production dependencies:
git clone [https://github.com/syed-omer-123/youtube-rag-engine.git](https://github.com/syed-omer-123/youtube-rag-engine.git)
cd youtube-rag-engine
pip install -r requirements.txt

2. Environment Setup
Create a local .env file in the root directory and securely add your OpenAI credential key:
OPENAI_API_KEY=your_secret_openai_api_key_here

3. Execution
Launch the interactive production dashboard locally:
streamlit run app.py

🛡️ Production Hygiene
Zero Hallucinations: Rigorously tested against out-of-scope queries; prefers safe structural refusal over content generation.

Data Security: The .env configuration layer is completely isolated from version control tracking via strict workspace configuration rules.

Budget Constrained: Fully compatible with OpenAI's prepaid usage limits to eliminate runaway system API overhead.
