"""
Query page for the Streamlit UI.
"""

import streamlit as st
import time
from src.ui.components import header, metadata_filters, api_request, display_document, loading_spinner


def render_query_page():
    """Render the query page."""
    header(
        "Query Automotive Specifications",
        "Ask questions about automotive specifications and get accurate answers."
    )
    
    # Query input
    if "current_query" in st.session_state:
        query = st.text_area("Enter your question", value=st.session_state.current_query, height=100)
        # Clear the current query from session state to avoid persisting it
        st.session_state.current_query = ""
    else:
        query = st.text_area("Enter your question", height=100)
    
    # Metadata filters (expandable section)
    with st.expander("Advanced Filters"):
        filter_data = metadata_filters()
    
    # Top-K slider
    top_k = st.slider("Maximum number of documents to retrieve", min_value=1, max_value=20, value=5)
    
    # Submit button
    if st.button("Submit Query", type="primary"):
        if not query:
            st.warning("Please enter a query")
            return
            
        process_query(query, filter_data, top_k)


def process_query(query: str, metadata_filter, top_k: int = 5):
    """Process a query and display results."""
    # Prepare request data
    request_data = {
        "query": query,
        "metadata_filter": metadata_filter,
        "top_k": top_k
    }
    
    # Track timing
    start_time = time.time()
    
    # Make API request with loading spinner
    with loading_spinner("Processing your query..."):
        result = api_request(
            endpoint="/query/",
            method="POST",
            data=request_data
        )
    
    if not result:
        return
        
    # Display answer
    st.header("Answer")
    
    # Calculate total processing time
    processing_time = time.time() - start_time
    api_time = result.get("execution_time", 0)
    
    # Show answer with citations
    st.markdown(result["answer"])
    
    # Display timing information
    st.caption(f"Total processing time: {processing_time:.2f}s (API: {api_time:.2f}s)")
    
    # Display sources
    st.header("Sources")
    
    if "documents" in result and result["documents"]:
        for i, doc in enumerate(result["documents"]):
            display_document(doc, i)
    else:
        st.info("No source documents were retrieved for this query.")
        
    # Option to refine query
    st.subheader("Refine your query")
    
    # Suggest follow-up queries based on returned documents
    suggestions = generate_suggestions(query, result.get("documents", []))
    
    if suggestions:
        st.write("Try asking:")
        cols = st.columns(len(suggestions))
        
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    st.session_state.current_query = suggestion
                    st.experimental_rerun()


def generate_suggestions(query: str, documents: list) -> list:
    """Generate follow-up query suggestions based on the original query and results."""
    suggestions = []
    
    # Extract metadata from documents
    manufacturers = set()
    models = set()
    years = set()
    
    for doc in documents:
        metadata = doc.get("metadata", {})
        if "manufacturer" in metadata and metadata["manufacturer"]:
            manufacturers.add(metadata["manufacturer"])
        if "model" in metadata and metadata["model"]:
            models.add(metadata["model"])
        if "year" in metadata and metadata["year"]:
            years.add(str(metadata["year"]))
    
    # Generate suggestions based on the query and document metadata
    query_lower = query.lower()
    
    # If query is about specs
    if "specs" in query_lower or "specifications" in query_lower:
        for model in models:
            for year in years:
                suggestions.append(f"What are the dimensions of the {year} {model}?")
                break
            break
    
    # If query is about horsepower
    if "horsepower" in query_lower or "power" in query_lower:
        if models and years:
            model = next(iter(models))
            year = next(iter(years))
            suggestions.append(f"What is the torque of the {year} {model}?")
    
    # If query is about one model, suggest comparison
    if len(models) == 1 and manufacturers:
        model = next(iter(models))
        manufacturer = next(iter(manufacturers))
        
        # Find comparable models
        comparable = {
            "Camry": "Honda Accord",
            "Corolla": "Honda Civic",
            "RAV4": "Honda CR-V",
            "F-150": "Chevrolet Silverado",
            "Civic": "Toyota Corolla",
            "Accord": "Toyota Camry",
            "CR-V": "Toyota RAV4",
            "3 Series": "Mercedes C-Class",
            "Model 3": "BMW i4"
        }
        
        if model in comparable:
            suggestions.append(f"Compare {manufacturer} {model} with {comparable[model]}")
    
    # If query is about a specific feature
    features = ["safety", "fuel efficiency", "mpg", "entertainment", "navigation"]
    for feature in features:
        if feature in query_lower and models:
            model = next(iter(models))
            suggestions.append(f"What {feature} features does the {model} have?")
            break
            
    # Limit to 3 suggestions
    return suggestions[:3]
