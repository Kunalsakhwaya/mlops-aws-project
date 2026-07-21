import pandas as pd
import pickle
import os

# Paths
DATA_PATH = "data/ratings.csv"
MODEL_PATH = "model.pkl"

def train_model():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file {DATA_PATH} not found.")
        return
        
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
