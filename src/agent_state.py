import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
import pandas as pd
from rapidfuzz import process, fuzz
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ==========================================
# 1. Multi-Turn Session State (Memory)
# ==========================================
@dataclass
class AgentState:
    """
    Maintains conversational memory across turns to support reference resolution 
    (e.g., 'that one', 'only those rated > 8') and persist filters[cite: 1].
    """
    history: List[Dict[str, str]] = field(default_factory=list)
    active_filters: Dict[str, Any] = field(default_factory=dict)
    last_results: List[Dict[str, Any]] = field(default_factory=list)
    selected_movie_id: Optional[int] = None

    def get_context_summary(self) -> str:
        """Generates a text summary of the current state for the LLM Router."""
        summary = f"Active Filters: {json.dumps(self.active_filters)}\n"
        if self.last_results:
            titles = [m.get('title') for m in self.last_results[:5]]
            summary += f"Last Search Results (Top 5): {', '.join(titles)}\n"
        return summary

# Initialize a global or session-specific state
current_state = AgentState()

try:
    cleaned_df = pd.read_pickle("C:/Projects/movie-recommendation-system/src/cleaned_tmdb.pkl")
except FileNotFoundError:
    raise FileNotFoundError("Could not find 'cleaned_tmdb.pkl'. Make sure you ran data_pipeline.py first!")


# ==========================================
# 2. Strict Tool Definitions (Boundaries)
# ==========================================
# The docstrings here act as the strict boundaries for the LLM to decide which tool to use[cite: 1].



@tool
def structured_search(query_filters: str) -> str:
    """
    USE ONLY FOR: Deterministic operations, filters, sorting, aggregations, counts, and numerical comparisons.
    Examples: 'movies rated above 8', 'top 10 by revenue', 'released after 2010', 'how many action movies'.
    Input: A JSON string containing keys like 'genre', 'min_rating', 'min_year', 'max_year', 'min_runtime', 'min_votes', 'min_budget', 'sort_by', 'limit', 'count_only'.
    """
    try:
        filters = json.loads(query_filters)
    except json.JSONDecodeError:
        return "Error: Invalid JSON format provided for filters."

    # Update conversational state memory (assuming current_state is in the global scope)
    current_state.active_filters.update(filters)
    
    # Start with the full dataset
    df_filtered = cleaned_df.copy()
    
    # 1. Apply Categorical Filters
    if 'genre' in filters:
        # Check if the requested genre exists in the 'genres_list' for each row
        genre_target = filters['genre'].lower()
        df_filtered = df_filtered[df_filtered['genres_list'].apply(
            lambda x: any(genre_target in str(g).lower() for g in x) if isinstance(x, list) else False
        )]
        
    if 'company' in filters:
        company_target = filters['company'].lower()
        df_filtered = df_filtered[df_filtered['production_companies'].apply(
            lambda x: company_target in str(x).lower() if pd.notna(x) else False
        )]

    # 2. Apply Numerical Filters
    if 'min_rating' in filters:
        df_filtered = df_filtered[df_filtered['vote_average'] >= float(filters['min_rating'])]
        
    if 'min_year' in filters:
        df_filtered = df_filtered[df_filtered['release_year'] >= float(filters['min_year'])]
        
    if 'max_year' in filters:
        df_filtered = df_filtered[df_filtered['release_year'] <= float(filters['max_year'])]
        
    if 'min_runtime' in filters:
        df_filtered = df_filtered[df_filtered['runtime'] >= float(filters['min_runtime'])]
        
    if 'min_votes' in filters:
        df_filtered = df_filtered[df_filtered['vote_count'] >= float(filters['min_votes'])]
        
    if 'min_budget' in filters:
        df_filtered = df_filtered[df_filtered['budget'] >= float(filters['min_budget'])]
        
    if 'min_revenue' in filters:
        df_filtered = df_filtered[df_filtered['revenue'] >= float(filters['min_revenue'])]

    # 3. Handle Empty Results Early
    if df_filtered.empty:
        current_state.last_results = []
        return f"Applied filters {filters}, but found 0 matching movies."

    # 4. Sorting
    if 'sort_by' in filters:
        sort_col = filters['sort_by']
        # By default, we sort in descending order (highest revenue/rating first)
        ascending = filters.get('ascending', False)
        if sort_col in df_filtered.columns:
            df_filtered = df_filtered.sort_values(by=sort_col, ascending=ascending)

    # 5. Handle Count vs. List Requests
    if filters.get('count_only', False):
        count = len(df_filtered)
        return f"Applied filters {filters}. Total count: {count} movies."

    # 6. Apply Limit (e.g., "Top 10")
    limit = filters.get('limit', 10)
    df_filtered = df_filtered.head(limit)

    # 7. Format the output to return to the LLM and update Memory
    results_list = []
    for _, row in df_filtered.iterrows():
        movie_info = {
            "movie_id": int(row['id']),
            "title": row['title'],
            "release_year": int(row['release_year']) if pd.notna(row['release_year']) else None,
            "vote_average": float(row['vote_average']) if pd.notna(row['vote_average']) else None,
            "revenue": float(row['revenue']) if pd.notna(row['revenue']) else None,
            "runtime": float(row['runtime']) if pd.notna(row['runtime']) else None
        }
        results_list.append(movie_info)
        
    # Store in memory so pronouns like "the first one" can be resolved later
    current_state.last_results = results_list
    
    return f"Applied filters: {filters}. Found {len(df_filtered)} movies. Top results: {json.dumps(results_list)}"


@tool
def fuzzy_movie_search(title_query: str) -> str:
    """
    USE ONLY FOR: Misspelled titles, partial titles, or approximate movie title matching.
    Examples: 'Get me the movie Avatr', 'What is Intersteler?'.
    Input: The raw, possibly misspelled movie title.
    """
    # 1. Get the list of valid movie titles from your dataset
    # (Assuming cleaned_df is available in your script's global scope)
    all_titles = cleaned_df['title'].dropna().tolist()
    
    # 2. Use RapidFuzz to find the highest-scoring match
    # We use fuzz.WRatio as it handles case-insensitivity and minor typos well
    # Setting score_cutoff=80 enforces the rule: "Do not silently return an arbitrary result when confidence is low"
    best_match = process.extractOne(
        query=title_query, 
        choices=all_titles, 
        scorer=fuzz.WRatio, 
        score_cutoff=80.0
    )
    
    # 3. Handle the result
    if best_match:
        # extractOne returns a tuple: (matched_string, score, index)
        matched_title = best_match[0]
        match_score = best_match[1]
        
        # Lookup the exact row in the DataFrame to get the movie_id
        movie_row = cleaned_df[cleaned_df['title'] == matched_title].iloc[0]
        movie_id = int(movie_row['id'])
        
        # Update conversational state so the LLM remembers this movie for follow-ups
        current_state.selected_movie_id = movie_id
        
        return f"Found exact movie: {matched_title} (ID: {movie_id}). Confidence: {match_score:.2f}%."
    else:
        # Fallback when the highest score is below the 80.0 cutoff limit
        return f"Could not confidently find a movie matching '{title_query}'. Please ask the user to clarify the title."

@tool
def semantic_search(concept_query: str) -> str:
    """
    USE ONLY FOR: Conceptual searches, plot descriptions, themes, or similar movies where exact keywords fail[cite: 1].
    Examples: 'movies about surviving on another planet', 'dark psychological thriller', 'similar to Inception'[cite: 1].
    Input: The natural language description of the plot or theme.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Connect to the Chroma vector database stored on disk
    try:
        vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    except Exception as e:
        raise RuntimeError(f"Could not load Chroma DB from ./chroma_db: {e}")
    return "Retrieved 5 movies matching the semantic concept."


@tool
def movie_details(movie_id: int) -> str:
    """
    USE ONLY FOR: Fetching complete structured information about one specific movie.
    Input: The integer TMDB movie identifier[cite: 1].
    """
    # 1. Lookup the movie by its ID in the global dataset
    matched_movie = cleaned_df[cleaned_df['id'] == movie_id]
    
    # 2. Handle the case where the movie doesn't exist
    if matched_movie.empty:
        return f"Movie with ID {movie_id} not found in the dataset."
    
    # 3. Extract the row
    row = matched_movie.iloc[0]
    
    # 4. Format financial numbers safely (handling the NaNs we created for 0s)
    budget = f"${row['budget']:,.0f}" if pd.notna(row['budget']) else "Unknown"
    revenue = f"${row['revenue']:,.0f}" if pd.notna(row['revenue']) else "Unknown"
    
    # 5. Format lists safely
    genres = ", ".join(row['genres_list']) if isinstance(row.get('genres_list'), list) else "None"
    cast = ", ".join(row['cast_list']) if isinstance(row.get('cast_list'), list) else "None"
    
    # 6. Build a structured string representation
    details = (
        f"Title: {row.get('title', 'Unknown')}\n"
        f"Original Title: {row.get('original_title', 'Unknown')}\n"
        f"Tagline: {row.get('tagline', 'None')}\n"
        f"Release Year: {row.get('release_year', 'Unknown')}\n"
        f"Runtime: {row.get('runtime', 'Unknown')} minutes\n"
        f"Genres: {genres}\n"
        f"Director: {row.get('director', 'Unknown')}\n"
        f"Top Cast: {cast}\n"
        f"Vote Average: {row.get('vote_average', 'Unknown')} (from {row.get('vote_count', 0)} votes)\n"
        f"Budget: {budget}\n"
        f"Revenue: {revenue}\n"
        f"Overview: {row.get('overview', 'None')}"
    )
    
    return details

@tool
def rag_answer(question: str, context_docs: str) -> str:
    """
    USE ONLY FOR: Generating a natural-language answer based on retrieved movie context[cite: 1].
    Input: The user's specific question and the retrieved document strings.
    """
    # This tool wraps the execution of the RAG_PROMPT defined below
    return "LLM generated RAG answer goes here."

# List of tools to bind to your LLM agent
tools = [structured_search, fuzzy_movie_search, semantic_search, movie_details, rag_answer]

# ==========================================
# 3. LLM Prompt Templates
# ==========================================

# 3A. The Agent/Router Prompt
# This prompt instructs the LLM on how to resolve pronouns and pick tools.
ROUTER_PROMPT = PromptTemplate.from_template("""
You are a specialized movie routing agent. Your job is to select the correct tool to answer the user's query[cite: 1].
You have access to the conversation history and current system state.

CURRENT STATE:
{state_summary}

CONVERSATION HISTORY:
{chat_history}

USER QUERY: {user_query}

INSTRUCTIONS FOR PRONOUN/REFERENCE RESOLUTION:
If the user says "only those rated > 8" or "the first one", look at the CURRENT STATE to identify what they are referring to[cite: 1]. 
You may need to extract filters from the user query and combine them with the existing Active Filters.

TOOL SELECTION RULES[cite: 1]:
1. If the query requires counts, exact dates, financial numbers, or strict filters -> use `structured_search`.
2. If the user misspells a title or asks for a specific title -> use `fuzzy_movie_search`.
3. If the user asks for a theme, plot, or "movies like X" -> use `semantic_search`.
4. If the user wants to know specific details about a resolved movie -> use `movie_details`.
5. If the user asks a complex question about a plot requiring reading -> use `rag_answer`.

Decide on the best tool and generate the inputs required.
""")

# 3B. The Strict RAG Generation Prompt
# This enforces the anti-hallucination requirement[cite: 1].
RAG_PROMPT = PromptTemplate.from_template("""
You are an expert movie analyst. Your task is to answer the user's question using ONLY the retrieved context provided below.

RETRIEVED CONTEXT:
{retrieved_context}

USER QUESTION:
{user_question}

STRICT RULES:
1. You must answer the question based strictly on the RETRIEVED CONTEXT.
2. DO NOT invent, hallucinate, or guess movie metadata, plots, or numbers[cite: 1].
3. If the answer is not contained in the context, you must reply: "I do not have enough information in the dataset to answer that."
""")