import streamlit as st
from pipeline import Pipeline
from agents.cache import SQLiteCache
import json
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="MediSearch AI", layout="wide", page_icon="🧬")

# Custom Styling for modern premium feel
st.markdown("""
    <style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #4338ca;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 MediSearch AI")
st.markdown("### Agentic RAG Q&A + Biomedical Hypothesis Generation")

# Initialize cache
cache = SQLiteCache()

# Sidebar for Settings
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    
    st.markdown("---")
    st.subheader("Retrieval Settings")
    chunk_size = st.number_input("Chunk size (words)", min_value=50, max_value=1000, value=300, step=50)
    top_k = st.number_input("Number of top chunks to retrieve", min_value=1, max_value=10, value=3)
    
    st.markdown("---")
    st.subheader("Metadata Filters")
    page_filter = st.number_input("Filter search to page (0 for all)", min_value=0, max_value=200, value=0, step=1)
    
    st.markdown("---")
    st.subheader("System Performance & Caching")
    use_cache = st.checkbox("Enable Response Cache", value=True)
    force_reindex = st.checkbox("Force Reindex Embeddings", value=False)
    
    if st.button("Clear System Cache"):
        cache.clear()
        st.success("SQLite database cache cleared!")

# Main view layout
col_left, col_right = st.columns([1, 1])

with col_left:
    uploaded_pdf = st.file_uploader("Upload a medical research PDF", type=["pdf"])
    query = st.text_input("Enter your research question / query:")
    topic = st.text_input("Optional: Topic for hypothesis generation (defaults to query)")

if st.button("Run Pipeline"):
    if not uploaded_pdf:
        st.warning("Please upload a PDF file to proceed.")
    elif not query.strip():
        st.warning("Please enter a query to proceed.")
    else:
        # Save uploaded PDF temporarily
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        pipeline = Pipeline()
        filter_dict = {"page": page_filter} if page_filter > 0 else None

        with st.spinner("Executing RAG Pipeline across medical agents..."):
            result = pipeline.process_pdf_and_answer(
                pdf_path="temp_uploaded.pdf",
                query=query,
                topic_for_hypotheses=topic if topic.strip() else None,
                chunk_size=chunk_size,
                top_k=top_k,
                force_reindex=force_reindex,
                use_cache=use_cache,
                filter_dict=filter_dict,
            )

        st.success("Pipeline run completed!")

        # Tabs for Answer, Hypotheses, Sources, and Execution Metrics
        tab_answer, tab_hypotheses, tab_chunks, tab_metrics = st.tabs([
            "💬 Answer", 
            "💡 Hypotheses", 
            "📄 Retrieved Chunks", 
            "📊 Performance Metrics"
        ])

        with tab_answer:
            st.subheader("System Generated Evidence-Based Answer")
            st.markdown(f"> {result['answer']}")

        with tab_hypotheses:
            st.subheader("Generated Scientific Hypotheses")
            st.json(result["hypotheses"])

        with tab_chunks:
            st.subheader(f"Top {len(result['retrieved_chunks'])} Retrieved Context Chunks")
            for i, chunk in enumerate(result["retrieved_chunks"], start=1):
                st.markdown(f"**Chunk {i} Context:**")
                st.info(chunk)
                st.markdown("---")

        with tab_metrics:
            st.subheader("⏱️ Ingestion & LLM Latency Breakdown")
            metrics_data = result.get("metrics", {})
            if metrics_data:
                # Build dataframe for visualization
                df_metrics = pd.DataFrame({
                    "Pipeline Step": list(metrics_data.keys()),
                    "Duration (seconds)": list(metrics_data.values())
                })
                # Filter out the total duration from the bar chart for better comparison
                df_chart = df_metrics[df_metrics["Pipeline Step"] != "pipeline_total_sec"]
                
                fig = px.bar(
                    df_chart, 
                    x="Pipeline Step", 
                    y="Duration (seconds)", 
                    title="Latency per Pipeline Component",
                    labels={"Duration (seconds)": "Duration (seconds)"},
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show key summary cards
                m1, m2 = st.columns(2)
                m1.metric("Total Processing Time", f"{metrics_data.get('pipeline_total_sec', 0.0)}s")
                m2.metric("LLM RAG Generation Time", f"{metrics_data.get('rag_synthesis_sec', 0.0)}s")
            else:
                st.info("Metrics not recorded (Response was retrieved instantly from Cache)")

            st.subheader("📁 Saved Pipeline Artifacts")
            for key, path in result["paths"].items():
                st.text(f"{key}: {path}")

