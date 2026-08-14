import pandas as pd
import ast
import json
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ==========================================
# 1. JSON Parsing Helpers
# ==========================================
def extract_names(json_string, limit=None):
    """Parses JSON string of dictionaries and extracts the 'name' field[cite: 1]."""
    if pd.isna(json_string):
        return []
    try:
        # TMDB dataset often uses stringified Python lists, ast.literal_eval is safer
        items = ast.literal_eval(json_string)
        names = [item['name'] for item in items]
        return names[:limit] if limit else names
    except (ValueError, SyntaxError):
        return []

def extract_director(crew_string):
    """Extracts the director's name from the JSON-encoded crew information[cite: 1]."""
    if pd.isna(crew_string):
        return None
    try:
        crew = ast.literal_eval(crew_string)
        for member in crew:
            if member.get('job') == 'Director':
                return member.get('name')
        return None
    except (ValueError, SyntaxError):
        return None

# ==========================================
# 2. Data Loading and Cleaning
# ==========================================
def load_and_preprocess_data(movies_path, credits_path):
    # Load CSV files
    movies_df = pd.read_csv(movies_path)
    credits_df = pd.read_csv(credits_path)
    
    # DROP the duplicate 'title' column from credits to avoid _x and _y suffixes
    if 'title' in credits_df.columns:
        credits_df = credits_df.drop(columns=['title'])
    
    # Join the two tables using movies.id = credits.movie_id[cite: 1]
    df = pd.merge(movies_df, credits_df, left_on='id', right_on='movie_id')
    
    # Parse JSON fields into usable structures[cite: 1]
    df['genres_list'] = df['genres'].apply(extract_names)
    df['keywords_list'] = df['keywords'].apply(extract_names)
    
    # Extract top 5 cast members to avoid context window overload
    df['cast_list'] = df['cast'].apply(lambda x: extract_names(x, limit=5))
    df['director'] = df['crew'].apply(extract_director)
    
    # Convert release_date to a proper date representation[cite: 1]
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    
    # Treat budget=0 and revenue=0 appropriately (convert to NaN)[cite: 1]
    df['budget'] = df['budget'].replace(0, np.nan)
    df['revenue'] = df['revenue'].replace(0, np.nan)
    
    # Handle missing values explicitly[cite: 1]
    df['overview'] = df['overview'].fillna("")
    df['tagline'] = df['tagline'].fillna("")
    
    return df

# ==========================================
# 3. RAG Data Representation & Vector Setup
# ==========================================
def create_movie_documents(df):
    """Creates a textual representation for each movie for Semantic Retrieval[cite: 1]."""
    documents = []
    
    for _, row in df.iterrows():
        # Skip movies with no overview
        if not row['overview']:
            continue
            
        # Design an appropriate textual representation[cite: 1]
        content = (
            f"Title: {row['title']}\n"
            f"Tagline: {row['tagline']}\n"
            f"Genres: {', '.join(row['genres_list'])}\n"
            f"Keywords: {', '.join(row['keywords_list'])}\n"
            f"Director: {row['director']}\n"
            f"Top Cast: {', '.join(row['cast_list'])}\n"
            f"Overview: {row['overview']}"
        )
        
        # Keep essential metadata for filtering
        metadata = {
            "movie_id": int(row['id']),
            "title": row['title'],
            "release_year": int(row['release_year']) if pd.notna(row['release_year']) else -1,
            "vote_average": float(row['vote_average']) if pd.notna(row['vote_average']) else 0.0,
            "genres": ", ".join(row['genres_list'])
        }
        
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
        
    return documents

def initialize_vector_store(documents, persist_directory="./chroma_db"):
    """Embeds and stores the movie documents in a local Chroma vector database."""
    print(f"Embedding {len(documents)} movies... This may take a few minutes.")
    
    # Using a fast, local embedding model (or swap with OpenAIEmbeddings)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create and persist the vector store
    vector_store = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    print(f"Vector store successfully created at {persist_directory}")
    return vector_store

# ==========================================
# 4. Execution Block
# ==========================================
if __name__ == "__main__":
    # Paths to your downloaded TMDB dataset files
    MOVIES_CSV = "C:/Projects/movie-recommendation-system/data/tmdb_5000_movies.csv"
    CREDITS_CSV = "C:/Projects/movie-recommendation-system/data/tmdb_5000_credits.csv"
    
    # 1. Clean and Join
    print("Loading and cleaning data...")
    cleaned_df = load_and_preprocess_data(MOVIES_CSV, CREDITS_CSV)
    
    # 2. You can save this cleaned DataFrame for your Structured Search tool[cite: 1]
    cleaned_df.to_pickle("cleaned_tmdb.pkl")
    
    # 3. Create Text Documents
    print("Creating text documents for RAG...")
    docs = create_movie_documents(cleaned_df)
    
    # 4. Build Vector DB
    initialize_vector_store(docs)