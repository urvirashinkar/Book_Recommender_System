# 📚 Book Recommendation System - Streamlit App

A collaborative filtering-based book recommendation system that suggests books similar to the ones you like using K-Nearest Neighbors (KNN) algorithm.

## Features

- 🎯 **Get Personalized Recommendations**: Select any book and get similar book recommendations
- 📖 **Browse & Search**: Explore all available books with search functionality
- 📊 **View Statistics**: See system statistics and top-rated books
- ⚡ **Fast & Efficient**: Uses sparse matrices and KNN for quick recommendations
- 🎨 **User-Friendly Interface**: Beautiful Streamlit-based web application

## Project Structure

```
book_recommendation_system/
├── app.py                      # Main Streamlit application
├── preprocessor.py             # Data loading and preprocessing module
├── helper.py                   # Recommendation model and utility functions
├── requirements.txt            # Python dependencies
├── Books.csv                   # Books dataset
├── Users.csv                   # Users dataset
├── Ratings.csv                 # Ratings dataset
└── README.md                   # This file
```

## Installation

1. **Clone or navigate to the project directory**:

   ```bash
   cd book_recommendation_system
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Run the Streamlit app with:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## How It Works

### Data Processing

1. **Load Data**: Reads Books.csv, Users.csv, and Ratings.csv
2. **Clean Data**:
   - Removes duplicate books
   - Filters users with 200+ ratings
   - Filters books with 50+ ratings
   - Creates user-item matrix

3. **Build Model**: Trains KNN model on the user-item matrix

### Recommendation Algorithm

- **Algorithm**: K-Nearest Neighbors (KNN) with brute force
- **Distance Metric**: Euclidean distance on user rating vectors
- **Data Structure**: Sparse matrices for efficient computation
- **Logic**: Finds books with similar rating patterns to the selected book

### Key Statistics

- **Minimum User Ratings**: 200+ ratings per user
- **Minimum Book Ratings**: 50+ ratings per book
- **Matrix Type**: User-Item sparse matrix
- **Model**: NearestNeighbors from scikit-learn

## Module Documentation

### `preprocessor.py`

Functions for data loading and preprocessing:

- **`load_data()`**: Load CSV files
- **`clean_data()`**: Clean and filter data
- **`create_pivot_table()`**: Create user-item matrix

### `helper.py`

Functions for model building and recommendations:

- **`build_recommendation_model()`**: Train KNN model
- **`get_book_index()`**: Find book in the system
- **`recommend_books()`**: Get recommendations for a book
- **`get_available_books()`**: List all available books

### `app.py`

Main Streamlit application with three tabs:

1. **Get Recommendations**: Interactive recommendation interface
2. **Browse Books**: Search and explore all books
3. **Statistics**: System metrics and analytics

## Usage Guide

### Getting Recommendations

1. Go to the "Get Recommendations" tab
2. Select a book from the dropdown
3. Choose number of recommendations (1-10)
4. Click "Get Recommendations"
5. View similar books displayed in a card format

### Browsing Books

1. Go to the "Browse Books" tab
2. Optionally search for a book by title
3. Sort by alphabetical order or rating count
4. View the complete list of available books

### Viewing Statistics

1. Go to the "Statistics" tab
2. See system-wide metrics (total books, users, ratings)
3. View top 10 rated books
4. Check rating distribution

## Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library (KNN model)
- **scipy**: Scientific computing (sparse matrices)

## Performance Notes

- Data is cached on first load using `@st.cache_resource`
- Sparse matrices reduce memory usage
- KNN algorithm provides quick nearest neighbor lookups
- Model training is performed once at startup

## Troubleshooting

### "Books.csv not found"

Ensure all three CSV files (Books.csv, Users.csv, Ratings.csv) are in the same directory as app.py

### "Book not found in database"

Make sure the exact book title exists. Try using the Browse Books tab to find the correct title.

### Slow initial load

First load creates the model, which may take a few seconds. Subsequent loads are faster due to caching.

## Future Enhancements

- Content-based filtering option
- User ratings history
- Hybrid recommendation approach
- Book cover images integration
- Export recommendations to CSV

## License

This project is open source and available under the MIT License.

## Author

Created as a book recommendation system using collaborative filtering.
