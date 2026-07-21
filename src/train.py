import pandas as pd
import pickle
import os

DATA_PATH = "data/ratings.csv"
MODEL_PATH = "model.pkl"

DEFAULT_DATA = """user_id,movie_id,rating
1,101,5
1,102,4
2,101,3
2,103,5
3,102,4
3,103,4
4,101,5
"""

def train_model():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Data file {DATA_PATH} not found. Creating default dataset...")
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        with open(DATA_PATH, "w") as f:
            f.write(DEFAULT_DATA)
        
    df = pd.read_csv(DATA_PATH)
    
    print("Training model (Dummy Collaborative Filtering)...")
    movie_stats = df.groupby('movie_id').agg({'rating': ['mean', 'count']})
    movie_stats.columns = ['mean_rating', 'rating_count']
    top_movies = movie_stats.sort_values(by=['mean_rating', 'rating_count'], ascending=False).index.tolist()
    
    model_data = {
        'top_movies': top_movies,
        'user_history': df.groupby('user_id')['movie_id'].apply(list).to_dict()
    }
    
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
        
    print("Training complete! model.pkl is ready.")

if __name__ == "__main__":
    train_model()
