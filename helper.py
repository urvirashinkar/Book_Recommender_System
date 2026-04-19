"""
Helper functions for the book recommendation system.
Handles model training and recommendation logic using KNN.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def build_recommendation_model(book_pivot):
    """
    Build a KNN model for book recommendations.
    
    Args:
        book_pivot (pd.DataFrame): Pivot table with books as index and users as columns
    
    Returns:
        NearestNeighbors: Trained KNN model
    """
    # Convert to sparse matrix for efficiency
    book_sparse = csr_matrix(book_pivot)
    
    # Initialize and fit KNN model
    model = NearestNeighbors(algorithm='brute')
    model.fit(book_sparse)
    
    return model


def get_book_index(book_name, book_pivot):
    """
    Get the index of a book in the pivot table.
    
    Args:
        book_name (str): Name of the book to find
        book_pivot (pd.DataFrame): Pivot table with books as index
    
    Returns:
        int: Index of the book, or None if not found
    
    Raises:
        ValueError: If book not found in the pivot table
    """
    try:
        book_id = np.where(book_pivot.index == book_name)[0][0]
        return book_id
    except IndexError:
        raise ValueError(f"Book '{book_name}' not found in the database")


def recommend_books(book_name, book_pivot, model, n_recommendations=5):
    """
    Get book recommendations similar to the given book.
    
    Args:
        book_name (str): Name of the book to get recommendations for
        book_pivot (pd.DataFrame): Pivot table with books as index
        model (NearestNeighbors): Trained KNN model
        n_recommendations (int): Number of recommendations to return (including the input book)
    
    Returns:
        list: List of recommended book titles (excluding the input book)
    
    Raises:
        ValueError: If book not found in the database
    """
    book_id = get_book_index(book_name, book_pivot)
    
    # Get distances and indices of n_recommendations nearest neighbors
    distances, suggestions = model.kneighbors(
        book_pivot.iloc[book_id, :].values.reshape(1, -1),
        n_neighbors=n_recommendations
    )
    
    # Extract book titles and exclude the input book
    recommended_books = []
    
    # suggestions is a 2D array with shape (1, n_neighbors)
    # suggestions[0] gives us the indices for our single query
    suggestion_indices = suggestions[0]
    
    for idx in suggestion_indices:
        suggested_title = book_pivot.index[idx]
        if suggested_title != book_name:
            recommended_books.append(suggested_title)
    
    return recommended_books


def get_available_books(book_pivot):
    """
    Get list of all available books in the database.
    
    Args:
        book_pivot (pd.DataFrame): Pivot table with books as index
    
    Returns:
        list: Sorted list of book titles
    """
    return sorted(book_pivot.index.tolist())
