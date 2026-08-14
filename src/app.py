import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

# Import tools and state from your Day 2 agent file
from agent_state import tools, current_state, ROUTER_PROMPT

# Load environment variables
load_dotenv()

# ==============================================================================
# 1. Page Configuration & Layout
# ==============================================================================
st.set_page_config(
    page_title="TMDB CineAgent | Movie Recommendation & QA",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎬 TMDB CineAgent: Intelligent Movie Assistant")
st.caption("Deterministic querying, semantic vector search, and grounded RAG powered by Gemini & LangGraph.")

# ==============================================================================
# 2. Session State Initialization
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "trace_history" not in st.session_state:
    # Stores per-turn tool execution logs for inspection
    st.session_state.trace_history = []

# ==============================================================================
# 3. LLM Setup with Retry Logic (Section 3.2)
# ==============================================================================
@st.cache_resource
def get_llm():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ API Key not found! Please set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in your `.env` file.")
        st.stop()
    
    # max_retries=5 provides exponential backoff for 429 Resource Exhausted / Rate Limits
    return ChatGoogleGenerativeAI(model= "gemini-3.5-flash-lite", max_retries=5)


llm = get_llm()

# ==============================================================================
# 4. Sidebar: Conversational State & System Health
# ==============================================================================
with st.sidebar:
    st.header("🧠 Live Agent State")
    st.markdown(f"**Selected Movie ID:** `{current_state.selected_movie_id or 'None'}`")
    
    st.subheader("Active Filters")
    if current_state.active_filters:
        st.json(current_state.active_filters)
    else:
        st.info("No active filters applied.")
    
    st.subheader("Last Retrieved Entities")
    if current_state.last_results:
        st.write(f"{len(current_state.last_results)} items in cache.")
        with st.expander("View Cached Metadata"):
            st.write([item.get('title', item.get('movie_id')) for item in current_state.last_results])
    else:
        st.text("Context cache empty.")
        
    st.divider()
    if st.button("🧹 Clear Conversation & State", use_container_width=True):
        st.session_state.messages = []
        st.session_state.trace_history = []
        current_state.history = []
        current_state.active_filters = {}
        current_state.selected_movie_id = None
        current_state.last_results = []
        st.rerun()

# ==============================================================================
# 5. Chat History Rendering with Execution Traces & Data Tables
# ==============================================================================
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render execution trace if present for this turn
        if msg["role"] == "assistant" and "trace" in msg and msg["trace"]:
            trace = msg["trace"]
            with st.expander("🔍 **Execution Trace & Tool Diagnostics**", expanded=False):
                # 1. Selected Tools
                tools_used = trace.get("tools_called", [])
                st.markdown(f"**Tools Activated:** `{', '.join(tools_used) if tools_used else 'Direct Synthesis (No Tool)'}`")
                
                # 2. Applied Filters
                if trace.get("filters"):
                    st.markdown("**Applied Filters:**")
                    st.json(trace["filters"])
                
                # 3. Fuzzy Matches & Clarifications
                if trace.get("fuzzy_log"):
                    st.markdown(f"**Fuzzy Match Log:** `{trace['fuzzy_log']}`")
                
                # 4. Expandable Retrieved RAG Context Chunks
                if trace.get("rag_context"):
                    st.markdown("**Retrieved Vector Documents (RAG Chunks):**")
                    for doc_idx, chunk in enumerate(trace["rag_context"]):
                        with st.expander(f"Chunk #{doc_idx+1}: {chunk.get('title', 'Document')}"):
                            st.caption(f"Metadata: Release {chunk.get('release_year')} | Genres: {chunk.get('genres')}")
                            st.text(chunk.get("content", ""))

        # Render structured table if present
        if "table_data" in msg and msg["table_data"] is not None:
            st.markdown("### 📊 Retrieved Results")
            st.dataframe(msg["table_data"], use_container_width=True, hide_index=True)

# ==============================================================================
# 6. User Query Handling & Execution Loop
# ==============================================================================
if user_query := st.chat_input("Ask about movies, budgets, semantic concepts, or directors..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent, executing tools, and grounding response..."):
            trace_record = {
                "tools_called": [],
                "filters": None,
                "rag_context": None,
                "fuzzy_log": None
            }
            
            # Format dynamic system prompt with memory context
            formatted_prompt = ROUTER_PROMPT.format(
                state_summary=current_state.get_context_summary(),
                chat_history="Managed by LangGraph memory",
                user_query=user_query
            )
            
            agent_executor = create_agent(
                model=llm,
                tools=tools,
                system_prompt=formatted_prompt
            )
            
            # Append to message history
            current_state.history.append(HumanMessage(content=user_query))
            
            try:
                # Invoke the LangGraph execution loop
                response = agent_executor.invoke({"messages": current_state.history})
                
                # Parse tool execution intermediate steps from LangGraph messages
                for msg in response["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for call in msg.tool_calls:
                            tool_name = call.get("name")
                            if tool_name not in trace_record["tools_called"]:
                                trace_record["tools_called"].append(tool_name)
                            
                            # Extract structured search filters
                            if tool_name == "structured_search":
                                args = call.get("args", {})
                                raw_filters = args.get("query_filters")
                                if isinstance(raw_filters, str):
                                    try:
                                        trace_record["filters"] = json.loads(raw_filters)
                                    except:
                                        trace_record["filters"] = raw_filters
                                elif isinstance(raw_filters, dict):
                                    trace_record["filters"] = raw_filters
                            
                            # Extract fuzzy search logs
                            if tool_name == "fuzzy_movie_search":
                                trace_record["fuzzy_log"] = call.get("args")

                # Extract RAG chunks if semantic search was activated
                if "semantic_search" in trace_record["tools_called"] and current_state.last_results:
                    trace_record["rag_context"] = current_state.last_results

                # Extract final response content
                final_ai_msg = response["messages"][-1].content
                if isinstance(final_ai_msg, list):
                    answer_text = "".join([p.get("text", "") for p in final_ai_msg if isinstance(p, dict)])
                else:
                    answer_text = str(final_ai_msg)

                # Format structured table data if tabular records exist
                table_df = None
                if current_state.last_results and isinstance(current_state.last_results, list):
                    # Check if the results contain tabular movie metadata
                    if len(current_state.last_results) > 0 and "vote_average" in current_state.last_results[0]:
                        table_df = pd.DataFrame(current_state.last_results)
                        # Keep relevant columns clean for display
                        cols_to_show = [c for c in ["title", "release_year", "vote_average", "revenue", "runtime"] if c in table_df.columns]
                        table_df = table_df[cols_to_show]

                # Render assistant output
                st.markdown(answer_text)
                
                # Render live trace
                with st.expander("🔍 **Execution Trace & Tool Diagnostics**", expanded=False):
                    st.markdown(f"**Tools Activated:** `{', '.join(trace_record['tools_called']) if trace_record['tools_called'] else 'Direct Synthesis (No Tool)'}`")
                    if trace_record.get("filters"):
                        st.markdown("**Applied Filters:**")
                        st.json(trace_record["filters"])
                    if trace_record.get("rag_context"):
                        st.markdown("**Retrieved Vector Documents (RAG Chunks):**")
                        for doc_idx, chunk in enumerate(trace_record["rag_context"]):
                            with st.expander(f"Chunk #{doc_idx+1}: {chunk.get('title', 'Document')}"):
                                st.caption(f"Metadata: Release {chunk.get('release_year')} | Genres: {chunk.get('genres')}")
                                st.text(chunk.get("content", ""))

                if table_df is not None and not table_df.empty:
                    st.markdown("### 📊 Retrieved Results")
                    st.dataframe(table_df, use_container_width=True, hide_index=True)

                # Persist message to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "trace": trace_record,
                    "table_data": table_df
                })
                current_state.history.append(AIMessage(content=answer_text))

            except Exception as e:
                error_msg = f"⚠️ An error occurred during execution: `{str(e)}`"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})