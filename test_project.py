import pytest
import os
from project import MovieRecommendationSystem

@pytest.fixture(scope="module")
def recommender():
    return MovieRecommendationSystem(csv_file="data/imdb_top_1000.csv")

def test_data_loaded(recommender):
    assert recommender.df is not None
    assert len(recommender.df) > 0
    assert 'Series_Title' in recommender.df.columns

def test_content_based_preparation(recommender):
    assert recommender.tfidf_matrix is not None
    assert recommender.cosine_sim is not None
    assert recommender.tfidf_matrix.shape[0] == recommender.df.shape[0]
    assert recommender.cosine_sim.shape[0] == recommender.cosine_sim.shape[1]

def test_genre_column_exists(recommender):
    assert 'Genre' in recommender.df.columns
    assert recommender.df['Genre'].dropna().apply(lambda x: isinstance(x, str)).all()

def test_combined_features_column(recommender):
    assert 'combined_features' in recommender.df.columns
    assert recommender.df['combined_features'].apply(lambda x: isinstance(x, str)).all()

def test_analysis_image_created(recommender):
    # Manual run of analysis method
    recommender.analyze_dataset()
    assert os.path.exists("movie_analysis_professional.png")
