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

## 🚀 Installation & Setup

**1. Clone the repository and navigate to the project directory:**
```bash
git clone [https://github.com/yourusername/movie-recommendation-system.git](https://github.com/yourusername/movie-recommendation-system.git)
cd movie-recommendation-system