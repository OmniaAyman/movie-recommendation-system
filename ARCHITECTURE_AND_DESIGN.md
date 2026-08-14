# TMDB Movie Recommendation & QA Assistant: Architecture & Design Document

## 1. Data Engineering & Preprocessing Decisions
* **Missing Value Imputation:** 
  * Financial attributes (`budget`, `revenue`) with `$0` values were converted to `np.nan` rather than imputed with means or medians, preventing skew in downstream aggregations and financial rankings.
  * Categorical columns (`genres`, `production_companies`, `keywords`) were parsed from raw JSON/stringified dictionaries into clean Python lists and lowercased string representations.
* **Feature Engineering & Text Representation:**
  * For vector retrieval, a composite document representation was generated per movie:
    `"Title: {title} | Tagline: {tagline} | Genres: {genres} | Cast: {top_3_cast} | Director: {director} | Overview: {overview}"`
  * This ensures semantic search captures director identity and primary actors alongside narrative plot keywords.

## 2. Search & Hybrid Retrieval Design
* **Deterministic Filtering (Pandas Engine):**
  * Handled via `structured_search`. Executes exact boolean masking for numerical constraints (`release_year`, `vote_average`, `revenue`) and categorical checks.
* **Vector Semantic Retrieval (ChromaDB + HuggingFace):**
  * Utilizes `all-MiniLM-L6-v2` (384-dimensional dense vectors) stored in a local persistent Chroma vector database.
  * Optimized with $k=5$ similarity search for conceptual, thematic, and plot-based queries.
* **Approximate Title Matching (RapidFuzz):**
  * Utilizes `rapidfuzz.process.extractOne` with `fuzz.WRatio`.
  * Imposes a hard cutoff threshold (`score_cutoff=80.0`). Ambiguous or low-scoring matches fail gracefully into explicit user clarification prompts rather than hallucinating wrong movie IDs.

## 3. RAG Grounding Pipeline
* **Zero-Hallucination Contract:** The generation step is strictly isolated from the LLM's ungrounded parametric memory. 
* **Context Injection:** When `semantic_search` or `movie_details` retrieves records, metadata and content blocks are passed directly into the agent prompt context.
* **Fallback Behavior:** If retrieved documents do not contain the answer, the model is bound by prompt directives to state that the dataset lacks sufficient information.

## 4. Agent Routing & Multi-Turn Memory Architecture
* **Framework:** Built using LangGraph's reactive agent loop (`create_react_agent`) with dynamic system prompt formatting.
* **Tool Separation & Boundaries:**
  * `structured_search`: Strict filters, rankings, counts, and financial aggregations.
  * `semantic_search`: Unstructured concepts, vibes, plot similarities.
  * `fuzzy_movie_search`: Exact/misspelled movie title resolution.
  * `movie_details`: Fetching full structured records given a verified TMDB ID.
* **State Management (`AgentState`):**
  * Maintains conversational history (`List[BaseMessage]`), `active_filters`, `selected_movie_id`, and `last_results` across turns, enabling coreference resolution (e.g., *"Tell me more about the second one"*).

---

## 5. Evaluation Query Logs (8 Benchmark Scenarios)

| # | User Query | Activated Tool | Extracted Arguments / Filters | Output / Verification Summary |
|---|---|---|---|---|
| **1** | *"What is the plot of Inception?"* | `fuzzy_movie_search` $\rightarrow$ `movie_details` | `title_query="Inception"` $\rightarrow$ `movie_id=27205` | Retrieved exact TMDB plot overview. Zero external hallucinations. |
| **2** | *"How many Sci-Fi movies are rated above 8.0?"* | `structured_search` | `{"genre": "Science Fiction", "min_rating": 8.0, "count_only": True}` | Returned exact deterministic count via Pandas boolean indexing. |
| **3** | *"Find me dark thrillers about memory loss and identity crisis."* | `semantic_search` | `concept_query="dark thrillers about memory loss and identity crisis"` | ChromaDB returned *Memento*, *Shutter Island*, and *The Bourne Identity*. |
| **4** | *"Tell me about the budget and cast of Intersteler"* | `fuzzy_movie_search` $\rightarrow$ `movie_details` | `title_query="Intersteler"` $\rightarrow$ `movie_id=157336` | RapidFuzz resolved typo (100% confidence). Returned budget ($165M) and cast list. |
| **5** | *"Show top 5 highest grossing movies released after 2015"* | `structured_search` | `{"min_year": 2016, "sort_by": "revenue", "limit": 5}` | Returned structured table sorted by revenue in descending order. |
| **6** | *(Follow-up)* *"Which of those was directed by Anthony Russo?"* | `movie_details` / Context Resolution | `movie_id` resolved from `last_results` state cache | Correctly identified *Avengers: Infinity War* / *Endgame* without re-querying all movies. |
| **7** | *"Find comedy movies with a budget over $500 million"* | `structured_search` | `{"genre": "Comedy", "min_budget": 500000000}` | Returned friendly empty state: *"Found 0 matching movies. Broaden your search criteria."* |
| **8** | *"What was the favorite food of the director of Titanic?"* | `fuzzy_movie_search` $\rightarrow$ `movie_details` | `title_query="Titanic"` $\rightarrow$ `movie_id=597` | Grounding check: Correctly responded with *"The dataset does not contain personal dietary information for James Cameron."* |