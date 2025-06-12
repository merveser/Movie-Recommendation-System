import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class MovieRecommendationSystem:
    def __init__(self, csv_file="data/imdb_top_1000.csv"):
        """Initialize the recommendation system with IMDB top 1000 movies data"""
        self.df = None
        self.tfidf_matrix = None
        self.cosine_sim = None

        # Load data
        self.load_data(csv_file)
        self.prepare_content_based()

    def load_data(self, csv_file):
        """Load and clean the movie dataset"""
        try:
            self.df = pd.read_csv(csv_file, encoding='utf-8')
            print(f"✅ Dataset loaded successfully! {len(self.df)} movies found.")

            # Clean data
            self.df = self.df.dropna(subset=['Series_Title', 'Genre', 'Overview'])
            self.df['IMDB_Rating'] = pd.to_numeric(self.df['IMDB_Rating'], errors='coerce')
            self.df['Released_Year'] = pd.to_numeric(self.df['Released_Year'], errors='coerce')
            self.df['No_of_Votes'] = pd.to_numeric(self.df['No_of_Votes'], errors='coerce')

            # Fill missing values
            self.df['Director'] = self.df['Director'].fillna('Unknown')
            self.df['Star1'] = self.df['Star1'].fillna('Unknown')
            self.df['Star2'] = self.df['Star2'].fillna('Unknown')
            self.df['Star3'] = self.df['Star3'].fillna('Unknown')
            self.df['Star4'] = self.df['Star4'].fillna('Unknown')

            print(f"✅ Data cleaned successfully! {len(self.df)} movies ready for recommendations.")

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def prepare_content_based(self):
        """Prepare TF-IDF matrix for content-based recommendations"""
        try:
            # Combine features for content similarity
            self.df['combined_features'] = (
                self.df['Genre'].astype(str) + ' ' +
                self.df['Overview'].astype(str) + ' ' +
                self.df['Director'].astype(str) + ' ' +
                self.df['Star1'].astype(str) + ' ' +
                self.df['Star2'].astype(str) + ' ' +
                self.df['Star3'].astype(str) + ' ' +
                self.df['Star4'].astype(str)
            )

            # Create TF-IDF matrix
            tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
            self.tfidf_matrix = tfidf.fit_transform(self.df['combined_features'])

            # Calculate cosine similarity matrix
            self.cosine_sim = cosine_similarity(self.tfidf_matrix)

            print("✅ Content-based recommendation system prepared!")

        except Exception as e:
            print(f"❌ Error preparing content-based system: {e}")

    def analyze_dataset(self):
        """Analyze and display dataset statistics with visualizations"""
        print("\n" + "="*50)
        print("MOVIE DATABASE ANALYSIS")
        print("="*50)

        print(f" Total Movies: {len(self.df)}")
        print(f" Year Range: {int(self.df['Released_Year'].min())} - {int(self.df['Released_Year'].max())}")
        print(f" Average IMDB Rating: {self.df['IMDB_Rating'].mean():.1f}/10")
        print(f" Unique Directors: {self.df['Director'].nunique()}")

        # Set matplotlib to non-interactive backend
        plt.ioff()

        try:
            # Define color palette
            pastel_blue = '#A8DADC'
            pastel_coral = '#F1FAEE'
            pastel_sage = '#457B9D'

            # Create figure
            plt.style.use('default')
            fig = plt.figure(figsize=(18, 12))
            fig.patch.set_facecolor('white')

            # Add main title
            fig.suptitle('Movie Database Analytics Dashboard',
                        fontsize=20, fontweight='bold', color='#2c3e50', y=0.96)

            # 1. Top 10 Genres
            print("\n Analyzing movie genres...")
            genre_counts = {}
            for genres in self.df['Genre'].dropna():
                for genre in genres.split(', '):
                    genre_counts[genre.strip()] = genre_counts.get(genre.strip(), 0) + 1

            sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            print("Top 10 Genres:")
            for i, (genre, count) in enumerate(sorted_genres, 1):
                percentage = (count / len(self.df)) * 100
                print(f"   {i:2d}. {genre}: {count} movies ({percentage:.1f}%)")

            ax1 = plt.subplot(2, 3, 1)
            genres, counts = zip(*sorted_genres)


            colors = [pastel_blue, pastel_coral, pastel_sage] * 4

            wedges, texts, autotexts = ax1.pie(counts, labels=genres, colors=colors[:len(genres)],
                                              autopct='%1.1f%%', startangle=90, pctdistance=0.85,
                                              textprops={'fontsize': 9})


            centre_circle = plt.Circle((0,0), 0.70, fc='white')
            ax1.add_artist(centre_circle)
            ax1.set_title('Genre Distribution', fontsize=12, fontweight='bold', pad=20)

            # 2. Top 10 Directors
            print("\n Top directors analysis...")
            director_counts = self.df[self.df['Director'] != 'Unknown']['Director'].value_counts().head(10)

            print("Top 10 Directors:")
            for i, (director, count) in enumerate(director_counts.items(), 1):
                print(f"   {i:2d}. {director}: {count} movies")

            ax2 = plt.subplot(2, 3, 2)
            y_pos = range(len(director_counts))
            bars = ax2.barh(y_pos, director_counts.values, color=pastel_sage)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([name.split()[-1] if len(name.split()) > 1 else name for name in director_counts.index])
            ax2.set_xlabel('Number of Movies')
            ax2.set_title('Most Prolific Directors', fontsize=12, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)

            # Add value labels on bars
            for i, (bar, count) in enumerate(zip(bars, director_counts.values)):
                ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{count}', ha='left', va='center', fontweight='bold')

            # 3. IMDB Rating Distribution
            print("\n Rating distribution analysis...")
            ax3 = plt.subplot(2, 3, 3)
            n, bins, patches = ax3.hist(self.df['IMDB_Rating'].dropna(), bins=15,
                                       alpha=0.8, color=pastel_blue, edgecolor='black', linewidth=1)

            ax3.set_title('IMDB Rating Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('IMDB Rating')
            ax3.set_ylabel('Number of Movies')
            ax3.axvline(self.df['IMDB_Rating'].mean(), color=pastel_sage, linestyle='--', linewidth=2,
                       label=f'Average: {self.df["IMDB_Rating"].mean():.1f}')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)

            # 4. Movies by Decade
            print("\n Decade-wise movie production...")
            year_counts = self.df['Released_Year'].value_counts().sort_index()
            decade_counts = {}
            for year, count in year_counts.items():
                if pd.notna(year):
                    decade = int(year // 10) * 10
                    decade_counts[f"{decade}s"] = decade_counts.get(f"{decade}s", 0) + count

            decades = sorted(decade_counts.keys())
            counts = [decade_counts[decade] for decade in decades]

            ax4 = plt.subplot(2, 3, 4)
            ax4.fill_between(range(len(decades)), counts, alpha=0.7, color=pastel_coral)
            ax4.plot(range(len(decades)), counts, color=pastel_sage, linewidth=3, marker='o', markersize=6)
            ax4.set_title('Movies by Decade', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Decades')
            ax4.set_ylabel('Number of Movies')
            ax4.set_xticks(range(len(decades)))
            ax4.set_xticklabels(decades, rotation=45)
            ax4.grid(True, alpha=0.3)


            for i, count in enumerate(counts):
                ax4.annotate(f'{count}', (i, count), textcoords="offset points",
                           xytext=(0,10), ha='center', fontweight='bold')

            # 5. Top Actors
            print("\n Top actors analysis...")
            all_actors = []
            for _, row in self.df.iterrows():
                actors = [row['Star1'], row['Star2'], row['Star3'], row['Star4']]
                for actor in actors:
                    if pd.notna(actor) and actor != 'Unknown':
                        all_actors.append(actor)

            actor_counts = Counter(all_actors).most_common(8)

            ax5 = plt.subplot(2, 3, 5)
            actors, counts = zip(*actor_counts)


            pie_colors = [pastel_blue, pastel_coral, pastel_sage] * 3

            wedges, texts, autotexts = ax5.pie(counts, labels=[actor.split()[-1] for actor in actors],
                                              colors=pie_colors[:len(actors)],
                                              autopct='%1.1f%%', startangle=90,
                                              textprops={'fontsize': 9})
            ax5.set_title('Most Frequent Actors', fontsize=12, fontweight='bold')

            # 6. Rating vs Votes Scatter Plot
            print("\n📊 Rating vs popularity correlation...")
            ax6 = plt.subplot(2, 3, 6)

            scatter = ax6.scatter(self.df['No_of_Votes'], self.df['IMDB_Rating'],
                                c=pastel_sage, alpha=0.6, s=30)
            ax6.set_title('Rating vs Vote Count', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Number of Votes')
            ax6.set_ylabel('IMDB Rating')
            ax6.set_xscale('log')
            ax6.grid(True, alpha=0.3)

            # Adjust layout
            plt.tight_layout()

            # Save plot
            plt.savefig('movie_data_analysis_visualizations.png', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(" Analysis charts created and saved as 'movie_analysis.png'")
            plt.close()

        except Exception as e:
            print(f"  Could not generate visualizations: {e}")
            print(" Continuing with text-based analysis...")

        # Additional text-based analysis
        print(f"\n Key Statistics:")

        # Top rated movies
        print("\n Top 5 Highest Rated Movies:")
        top_rated = self.df.nlargest(5, 'IMDB_Rating')[['Series_Title', 'IMDB_Rating', 'Released_Year']]
        for i, (_, movie) in enumerate(top_rated.iterrows(), 1):
            print(f"   {i}. {movie['Series_Title']} ({int(movie['Released_Year'])}) - {movie['IMDB_Rating']}")

        # Most popular actors
        print("\n Top 5 Most Frequent Actors:")
        all_actors = []
        for _, row in self.df.iterrows():
            actors = [row['Star1'], row['Star2'], row['Star3'], row['Star4']]
            for actor in actors:
                if pd.notna(actor) and actor != 'Unknown':
                    all_actors.append(actor)

        actor_counts = Counter(all_actors).most_common(5)
        for i, (actor, count) in enumerate(actor_counts, 1):
            print(f"   {i}. {actor}: {count} movies")

        # Summary statistics
        decade_counts = {}
        year_counts = self.df['Released_Year'].value_counts().sort_index()
        for year, count in year_counts.items():
            if pd.notna(year):
                decade = int(year // 10) * 10
                decade_counts[f"{decade}s"] = decade_counts.get(f"{decade}s", 0) + count

        print(f"\n📈 Summary:")
        print(f"   • Most productive decade: {max(decade_counts.items(), key=lambda x: x[1])[0]} ({max(decade_counts.values())} movies)")
        print(f"   • Highest rated movie: {top_rated.iloc[0]['Series_Title']} ({top_rated.iloc[0]['IMDB_Rating']})")
        print(f"   • Most common genre: {sorted_genres[0][0]} ({sorted_genres[0][1]} movies)")
        print(f"   • Most prolific director: {director_counts.index[0]} ({director_counts.iloc[0]} movies)")
        print(f"   • Most frequent actor: {actor_counts[0][0]} ({actor_counts[0][1]} movies)")

    def get_content_based_recommendations(self, movie_title, num_recommendations=10):
        """Get content-based recommendations for a movie"""
        try:
            # Find movie index
            movie_indices = self.df[self.df['Series_Title'].str.contains(movie_title, case=False, na=False)].index

            if len(movie_indices) == 0:
                print(f"❌ Movie '{movie_title}' not found!")
                return

            movie_idx = movie_indices[0]
            movie_name = self.df.iloc[movie_idx]['Series_Title']

            # Get similarity scores
            sim_scores = list(enumerate(self.cosine_sim[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:num_recommendations+1]  # Exclude the movie itself

            print(f"\n Content-based recommendations for '{movie_name}':")
            print("-" * 60)

            for i, (idx, score) in enumerate(sim_scores, 1):
                movie = self.df.iloc[idx]
                print(f"{i:2d}. {movie['Series_Title']} ({int(movie['Released_Year'])}) - {movie['IMDB_Rating']}")
                print(f"     {movie['Genre']}")
                print(f"     Similarity: {score:.3f}")
                print()

        except Exception as e:
            print(f"❌ Error getting recommendations: {e}")

    def get_popular_recommendations(self, num_recommendations=10):
        """Get popular movie recommendations based on IMDB rating and vote count"""
        print(f"\n Top {num_recommendations} Popular Movies:")
        print("-" * 60)

        # Calculate weighted rating (similar to IMDB's weighted rating)
        min_votes = self.df['No_of_Votes'].quantile(0.8)  # Movies with votes in top 20%
        popular_movies = self.df[self.df['No_of_Votes'] >= min_votes]

        # Sort by IMDB rating
        popular_movies = popular_movies.nlargest(num_recommendations, 'IMDB_Rating')

        for i, (_, movie) in enumerate(popular_movies.iterrows(), 1):
            print(f"{i:2d}. {movie['Series_Title']} ({int(movie['Released_Year'])}) - {movie['IMDB_Rating']}")
            print(f"     {movie['Genre']}")
            print(f"     {movie['No_of_Votes']:,} votes")
            print()

    def get_recommendations_by_actor(self, actor_name, num_recommendations=10):
        """Get recommendations based on actor"""
        try:
            # Find movies with this actor
            actor_movies = self.df[
                (self.df['Star1'].str.contains(actor_name, case=False, na=False)) |
                (self.df['Star2'].str.contains(actor_name, case=False, na=False)) |
                (self.df['Star3'].str.contains(actor_name, case=False, na=False)) |
                (self.df['Star4'].str.contains(actor_name, case=False, na=False))
            ]

            if len(actor_movies) == 0:
                print(f"❌ No movies found with actor '{actor_name}'!")
                return

            # Sort by IMDB rating
            actor_movies = actor_movies.nlargest(num_recommendations, 'IMDB_Rating')

            print(f"\n Top {len(actor_movies)} movies with '{actor_name}':")
            print("-" * 60)

            for i, (_, movie) in enumerate(actor_movies.iterrows(), 1):
                print(f"{i:2d}. {movie['Series_Title']} ({int(movie['Released_Year'])}) - ⭐{movie['IMDB_Rating']}")
                print(f"     {movie['Genre']}")
                print(f"     Director: {movie['Director']}")
                print()

        except Exception as e:
            print(f"❌ Error getting actor recommendations: {e}")

    def get_recommendations_by_director(self, director_name, num_recommendations=10):
        """Get recommendations based on director"""
        try:
            # Find movies by this director
            director_movies = self.df[self.df['Director'].str.contains(director_name, case=False, na=False)]

            if len(director_movies) == 0:
                print(f"❌ No movies found by director '{director_name}'!")
                return

            # Sort by IMDB rating
            director_movies = director_movies.nlargest(num_recommendations, 'IMDB_Rating')

            print(f"\n Top {len(director_movies)} movies by '{director_name}':")
            print("-" * 60)

            for i, (_, movie) in enumerate(director_movies.iterrows(), 1):
                print(f"{i:2d}. {movie['Series_Title']} ({int(movie['Released_Year'])}) - ⭐{movie['IMDB_Rating']}")
                print(f"     {movie['Genre']}")
                print(f"     Stars: {movie['Star1']}, {movie['Star2']}")
                print()

        except Exception as e:
            print(f"❌ Error getting director recommendations: {e}")

    def search_content(self, search_term):
        """Search for movies by title, genre, or overview"""
        try:
            # Search in multiple fields
            results = self.df[
                (self.df['Series_Title'].str.contains(search_term, case=False, na=False)) |
                (self.df['Genre'].str.contains(search_term, case=False, na=False)) |
                (self.df['Overview'].str.contains(search_term, case=False, na=False)) |
                (self.df['Director'].str.contains(search_term, case=False, na=False))
            ]

            if len(results) == 0:
                print(f"❌ No movies found matching '{search_term}'!")
                return

            # Sort by IMDB rating
            results = results.nlargest(min(20, len(results)), 'IMDB_Rating')

            print(f"\n Search results for '{search_term}' ({len(results)} found):")
            print("-" * 60)

            for i, (_, movie) in enumerate(results.iterrows(), 1):
                print(f"{i:2d}. {movie['Series_Title']} ({int(movie['Released_Year'])}) - ⭐{movie['IMDB_Rating']}")
                print(f"     {movie['Genre']}")
                print(f"     Director: {movie['Director']}")
                if len(movie['Overview']) > 100:
                    print(f"     {movie['Overview'][:100]}...")
                else:
                    print(f"     {movie['Overview']}")
                print()

        except Exception as e:
            print(f"❌ Error searching: {e}")


def main():
    print(" Welcome to IMDB Movie Recommendation System!")
    print("=" * 50)

    try:
        recommender = MovieRecommendationSystem()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("Make sure the file 'data/imdb_top_1000.csv' exists in the correct location.")
        return

    while True:
        print("\n" + "=" * 50)
        print("🎭MOVIE RECOMMENDATION SYSTEM")
        print("=" * 50)
        print("1.  Analyze Dataset")
        print("2.  Get Content-Based Recommendations")
        print("3.  Get Popular Recommendations")
        print("4.  Get Recommendations by Actor")
        print("5.  Get Recommendations by Director")
        print("6.  Search Content")
        print("7.  Exit")
        print("-" * 50)

        try:
            choice = input("👉 Select an option (1-7): ").strip()

            if choice == '1':
                recommender.analyze_dataset()

            elif choice == '2':
                movie_title = input(" Enter movie title: ").strip()
                if movie_title:
                    recommender.get_content_based_recommendations(movie_title)

            elif choice == '3':
                recommender.get_popular_recommendations()

            elif choice == '4':
                actor_name = input(" Enter actor name: ").strip()
                if actor_name:
                    recommender.get_recommendations_by_actor(actor_name)

            elif choice == '5':
                director_name = input(" Enter director name: ").strip()
                if director_name:
                    recommender.get_recommendations_by_director(director_name)

            elif choice == '6':
                search_term = input(" Enter search term: ").strip()
                if search_term:
                    recommender.search_content(search_term)

            elif choice == '7':
                print("👋 Thank you for using IMDB Movie Recommendation System!")
                print(" Happy watching!")
                break

            else:
                print("❌ Invalid option! Please select 1-7.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

        input("\n Press Enter to continue...")


if __name__ == "__main__":
    main()
