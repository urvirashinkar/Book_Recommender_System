"""
Data preprocessing module for the book recommendation system.
Handles loading and cleaning data from CSV files.
"""

import pandas as pd
import numpy as np


def load_data(books_path='Books.csv', users_path='Users.csv', ratings_path='Ratings.csv'):
    """
    Load books, users, and ratings data from CSV files.
    
    Args:
        books_path (str): Path to Books.csv
        users_path (str): Path to Users.csv
        ratings_path (str): Path to Ratings.csv
    
    Returns:
        tuple: (books_df, users_df, ratings_df)
    """
    books = pd.read_csv(books_path, sep=';', encoding='latin-1', on_bad_lines='skip')
    users = pd.read_csv(users_path, sep=';', encoding='latin-1', on_bad_lines='skip')
    ratings = pd.read_csv(ratings_path, sep=';', encoding='latin-1', on_bad_lines='skip')
    
    return books, users, ratings


def clean_data(books, users, ratings, min_user_ratings=200, min_book_ratings=50):
    """
    Clean and preprocess the data.
    
    Args:
        books (pd.DataFrame): Books dataframe
        users (pd.DataFrame): Users dataframe
        ratings (pd.DataFrame): Ratings dataframe
        min_user_ratings (int): Minimum ratings per user to keep
        min_book_ratings (int): Minimum ratings per book to keep
    
    Returns:
        tuple: (books_clean, ratings_clean, merged_data)
    """
    # Remove duplicates from books
    books = books.drop_duplicates()
    
    # Filter users with at least min_user_ratings ratings
    user_rating_counts = ratings['User-ID'].value_counts()
    active_users = user_rating_counts[user_rating_counts >= min_user_ratings].index
    ratings = ratings[ratings['User-ID'].isin(active_users)]
    
    # Merge ratings with books
    ratings_with_books = ratings.merge(books, on='ISBN')
    
    # Count ratings per book
    book_rating_counts = ratings_with_books.groupby('Title')['Rating'].count().reset_index()
    book_rating_counts.rename(columns={'Rating': 'number_of_ratings'}, inplace=True)
    
    # Merge rating counts back
    final_rating = ratings_with_books.merge(book_rating_counts, on='Title')
    
    # Filter books with at least min_book_ratings ratings
    final_rating = final_rating[final_rating['number_of_ratings'] >= min_book_ratings]
    
    # Remove duplicate user-book pairs
    final_rating = final_rating.drop_duplicates(['User-ID', 'Title'])
    
    return books, final_rating


def create_pivot_table(final_rating):
    """
    Create a pivot table (user-item matrix) for collaborative filtering.
    
    Args:
        final_rating (pd.DataFrame): Cleaned and merged rating data
    
    Returns:
        pd.DataFrame: Pivot table with books as index and users as columns
    """
    book_pivot = final_rating.pivot_table(
        columns='User-ID',
        index='Title',
        values='Rating'
    )
    
    # Fill NaN values with 0
    book_pivot = book_pivot.fillna(0)
    
    return book_pivot
