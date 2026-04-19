"""
Streamlit app for the Book Recommendation System.
Uses collaborative filtering with K-Nearest Neighbors to recommend similar books.
"""

import streamlit as st
import pandas as pd
from preprocessor import load_data, clean_data, create_pivot_table
from helper import build_recommendation_model, recommend_books, get_available_books


# Page configuration
st.set_page_config(
    page_title="📚 Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .book-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_prepare_data():
    """Load and preprocess data with caching for performance."""
    try:
        # Load data
        books, users, ratings = load_data()
        
        # Clean data
        books_clean, final_rating = clean_data(books, users, ratings)
        
        # Create pivot table
        book_pivot = create_pivot_table(final_rating)
        
        # Build model
        model = build_recommendation_model(book_pivot)
        
        return book_pivot, model, books_clean, final_rating
    except FileNotFoundError as e:
        st.error(f"Error loading data files: {e}")
        st.info("Please ensure Books.csv, Users.csv, and Ratings.csv are in the same directory as app.py")
        return None, None, None, None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown("# 📚 Book Recommendation System")
    st.markdown(
        "Discover your next favorite book using collaborative filtering powered by "
        "K-Nearest Neighbors (KNN) algorithm."
    )
    
    # Load data
    with st.spinner("Loading and preparing data..."):
        book_pivot, model, books_clean, final_rating = load_and_prepare_data()
    
    if book_pivot is None or model is None:
        st.stop()
    
    # Sidebar information
    with st.sidebar:
        st.markdown("### 📊 System Information")
        st.metric("Total Books in System", len(book_pivot))
        st.metric("Total Users", len(book_pivot.columns))
        st.metric("Total Ratings", len(final_rating))
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info(
            "This recommendation system uses collaborative filtering to suggest "
            "books similar to your selection. The model is trained on users with "
            "200+ ratings and books with 50+ ratings."
        )
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendations", "📖 Browse Books", "📈 Statistics"])
    
    with tab1:
        st.markdown("## Get Book Recommendations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Book selection
            available_books = get_available_books(book_pivot)
            selected_book = st.selectbox(
                "Select a book to get recommendations for:",
                available_books,
                help="Choose a book you like to discover similar books"
            )
        
        with col2:
            # Number of recommendations
            n_recs = st.slider(
                "Number of recommendations:",
                min_value=1,
                max_value=10,
                value=5,
                help="How many book recommendations to show"
            )
        
        # Get recommendations
        if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):
            try:
                recommendations = recommend_books(
                    selected_book,
                    book_pivot,
                    model,
                    n_recommendations=n_recs + 1  # +1 because input book is included
                )
                
                if recommendations:
                    st.success(f"Found {len(recommendations)} recommendations for '{selected_book}'")
                    
                    st.markdown("---")
                    st.markdown("### 📚 Recommended Books:")
                    
                    # Display recommendations
                    for idx, book in enumerate(recommendations[:n_recs], 1):
                        st.markdown(f"**{idx}. {book}**")
                else:
                    st.warning("No recommendations found for this book.")
                
            except ValueError as e:
                st.error(f"Error: {e}")
    
    with tab2:
        st.markdown("## Browse All Books")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_term = st.text_input(
                "Search for a book:",
                placeholder="Enter book title...",
                help="Filter books by partial title match"
            )
        
        with col2:
            sort_option = st.selectbox(
                "Sort by:",
                ["Alphabetical", "Rating Count"],
                help="How to sort the book list"
            )
        
        # Filter books
        available_books = get_available_books(book_pivot)
        
        if search_term:
            filtered_books = [
                book for book in available_books
                if search_term.lower() in book.lower()
            ]
        else:
            filtered_books = available_books
        
        # Display books
        if filtered_books:
            st.markdown(f"Found **{len(filtered_books)}** books")
            
            # Create a dataframe for better display
            books_data = []
            for book in filtered_books[:50]:  # Show first 50
                # Get rating count for this book
                rating_count = len(final_rating[final_rating['Title'] == book])
                books_data.append({
                    "Title": book,
                    "Rating Count": rating_count
                })
            
            books_df = pd.DataFrame(books_data)
            
            if sort_option == "Rating Count":
                books_df = books_df.sort_values("Rating Count", ascending=False)
            
            st.dataframe(books_df, use_container_width=True, hide_index=True)
        else:
            st.warning(f"No books found matching '{search_term}'")
    
    with tab3:
        st.markdown("## System Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Books",
                len(book_pivot),
                help="Books with 50+ ratings"
            )
        
        with col2:
            st.metric(
                "Total Users",
                len(book_pivot.columns),
                help="Users with 200+ ratings"
            )
        
        with col3:
            st.metric(
                "Total Ratings",
                len(final_rating),
                help="Valid ratings in the system"
            )
        
        # Show top rated books
        st.markdown("### 🌟 Top Rated Books")
        
        top_books = final_rating.groupby('Title')['Rating'].agg(['mean', 'count']).reset_index()
        top_books.columns = ['Title', 'Average Rating', 'Number of Ratings']
        top_books = top_books.sort_values('Average Rating', ascending=False).head(10)
        
        st.dataframe(top_books, use_container_width=True, hide_index=True)
        
        # Rating distribution
        st.markdown("### 📊 Rating Distribution")
        
        rating_dist = final_rating['Rating'].value_counts().sort_index()
        st.bar_chart(rating_dist)


if __name__ == "__main__":
    main()
