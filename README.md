# 🎬 TMDB CineAgent: Intelligent Movie Assistant

TMDB CineAgent is a multi-turn, intelligent movie recommendation and Q&A assistant built with **LangGraph**, **Streamlit**, and **Google Gemini**. It uses a robust agentic routing architecture to seamlessly switch between deterministic data filtering, semantic vector retrieval (RAG), and fuzzy string matching to provide accurate, grounded answers about movies.

## ✨ Features

* **Multi-Tool Agent Routing:** The LangGraph agent intelligently routes user queries to the most appropriate data tool:
  * **Structured Search (Pandas):** Handles deterministic operations like counts, financial aggregations, and strict filters (e.g., "Top 10 Action movies released after 2010").
  * **Semantic Search (ChromaDB + HuggingFace):** Executes vector-based conceptual and thematic searches (e.g., "Dark psychological thrillers about memory loss").
  * **Fuzzy Title Matching (RapidFuzz):** Gracefully handles typos and partial titles, preventing the LLM from hallucinating wrong movie IDs.
  * **Movie Details Extraction:** Retrieves precise, structured metadata directly from the dataset.
* **Multi-Turn Memory & State:** Remembers active filters, recent results, and selected movies, allowing for complex conversational coreferences (e.g., *"What was the budget of the second one?"*).
* **Zero-Hallucination Design:** Grounded strictly in the TMDB dataset. If the data isn't there, the agent admits it.
* **Transparent UI:** Built with Streamlit, featuring an expandable execution trace to visualize exactly which tools were called, what filters were applied, and what vector chunks were retrieved.

---

## 🛠️ Prerequisites

* Python 3.9+
* A valid Google Gemini API Key (Available for free at [Google AI Studio](https://aistudio.google.com/app/apikey))

---

## 🚀 Installation & Setup

Follow these steps to configure the environment, build the local vector database, and launch the application.

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/movie-recommendation-system.git](https://github.com/yourusername/movie-recommendation-system.git)
cd movie-recommendation-system
```

**2. Create and activate a virtual environment**
It is highly recommended to use a virtual environment to avoid dependency conflicts.
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
Install the required packages using the provided `requirements.txt` file.
```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables**
The application requires a Google Gemini API key to run the LangGraph agent. 
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Open the newly created `.env` file and replace the placeholder with your actual API key:
   ```env
   # Get a free API key at: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
   GEMINI_API_KEY=your_actual_api_key_here
   ```

**5. Run the Data Pipeline (Required once)**
Before starting the chat assistant, you must process the raw TMDB dataset and generate the local ChromaDB vector database.
```bash
python src/data_pipeline.py
```
*(Note: This script will clean the dataset, generate `cleaned_tmdb.pkl`, and build a local `./chroma_db` folder containing the vector embeddings. This may take 1–3 minutes depending on your hardware.)*

**6. Launch the Streamlit Application**
Once the data pipeline completes successfully, you can start the interactive chat interface!
```bash
streamlit run src/app.py
```
The application will automatically open in your default web browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
├── .env.example                 # Template for environment variables
├── .gitignore                   # Ignored files (including .env and datasets)
├── ARCHITECTURE_AND_DESIGN.md   # Detailed design decisions and query logs
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── src/
    ├── app.py                   # Streamlit UI & LangGraph execution loop
    ├── agent.py                 # Tool definitions, Memory State, and Prompts
    ├── cli_test.py              # Command-line interface for rapid testing
    └── data_pipeline.py         # Pandas cleaning & ChromaDB embedding logic
```

---

## 🏗️ Architecture details
For an in-depth breakdown of the data engineering decisions, RAG chunking strategy, tool boundaries, and evaluation query logs, please refer to the [ARCHITECTURE_AND_DESIGN.md](ARCHITECTURE_AND_DESIGN.md) document.