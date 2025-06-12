# IMDB Movie Recommendation System

A comprehensive Python-based movie recommendation system that analyzes the IMDB Top 1000 movies dataset and provides personalized movie recommendations using content-based filtering techniques.

## Features

### Core Functionality
- **Content-Based Recommendations**: Get movie suggestions based on genre, cast, director, and plot similarities
- **Popular Movie Recommendations**: Discover highly-rated movies with significant vote counts
- **Actor-Based Recommendations**: Find the best movies featuring your favorite actors
- **Director-Based Recommendations**: Explore top-rated films by specific directors
- **Smart Search**: Search movies by title, genre, plot, or director
- **Dataset Analysis**: Comprehensive statistical analysis with beautiful visualizations

### Data Analysis & Visualization
- Genre distribution analysis with interactive pie charts
- Director productivity rankings
- IMDB rating distribution histograms
- Movie production trends by decade
- Actor frequency analysis
- Rating vs popularity correlation plots

## Installation

### Prerequisites
Make sure you have Python 3.7+ installed on your system.

### Required Libraries
Install the required dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn pytest
```

### Dataset Setup
1. Create a `data` folder in your project directory
2. Place your `imdb_top_1000.csv` file in the `data` folder
3. Ensure your CSV file contains the following columns:
   - Series_Title
   - Genre
   - Overview
   - IMDB_Rating
   - Released_Year
   - Director
   - Star1, Star2, Star3, Star4
   - No_of_Votes

## Usage

### Running the Application
Simply run the main script:

```bash
python movie_recommendation_system.py
```

### Interactive Menu Options

**1. Analyze Dataset**
- Generates comprehensive statistics about your movie dataset
- Creates visualizations showing genre distributions, rating patterns, and trends
- Saves analysis charts as PNG files for future reference

**2. Get Content-Based Recommendations**
- Enter a movie title to get similar movie suggestions
- Uses TF-IDF vectorization and cosine similarity
- Considers genre, plot, director, and cast for recommendations

**3. Get Popular Recommendations**
- Shows top-rated movies with high vote counts
- Uses a weighted rating system similar to IMDB's methodology
- Perfect for discovering universally acclaimed films

**4. Get Recommendations by Actor**
- Enter an actor's name to see their best movies
- Results sorted by IMDB rating
- Great for exploring an actor's filmography

**5. Get Recommendations by Director**
- Find top movies by your favorite directors
- Ranked by critical acclaim and audience ratings

**6. Search Content**
- Flexible search across multiple fields
- Search by movie title, genre, plot keywords, or director
- Results ranked by relevance and rating

## How It Works

### Content-Based Filtering
The system uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to analyze movie features:

1. **Feature Combination**: Combines genre, plot overview, director, and cast information
2. **Text Processing**: Uses TF-IDF to convert text features into numerical vectors
3. **Similarity Calculation**: Computes cosine similarity between movies
4. **Recommendation Generation**: Suggests movies with highest similarity scores

### Data Processing
- Handles missing values gracefully
- Normalizes rating and vote data
- Cleans and preprocesses text features
- Optimizes performance with efficient data structures


## Sample Output

### Content-Based Recommendations
```
Content-based recommendations for 'The Dark Knight':
-----------------------------------------------------------
 1. Batman Begins (2005) - 8.2
    Action, Crime, Drama
    Similarity: 0.845

 2. The Dark Knight Rises (2012) - 8.4
    Action, Crime, Drama
    Similarity: 0.823
```

### Dataset Analysis
```
MOVIE DATABASE ANALYSIS
==================================================
Total Movies: 1000
Year Range: 1920 - 2020
Average IMDB Rating: 8.3/10
Unique Directors: 644

Top 10 Genres:
   1. Drama: 278 movies (27.8%)
   2. Action: 267 movies (26.7%)
   3. Crime: 150 movies (15.0%)
```

## Troubleshooting

### Common Issues
- **File Not Found**: Ensure `imdb_top_1000.csv` is in the `data` folder
- **Missing Dependencies**: Install all required libraries using pip
- **Encoding Issues**: The system handles UTF-8 encoding automatically
- **Memory Errors**: For very large datasets, consider increasing system memory

### Data Quality
The system automatically handles:
- Missing values in non-critical fields
- Invalid rating or year data
- Inconsistent text formatting
- Duplicate entries

## License

This project is open source and available under the MIT License.

---

**Enjoy the movie!** This system will help you discover your next favorite film based on your preferences and viewing history.
