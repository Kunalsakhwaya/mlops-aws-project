from fastapi import FastAPI, HTTPException
import pickle
import os

app = FastAPI(title="Movie Recommender API")

MODEL_PATH = "model.pkl"
model_data = None

@app.on_event("startup")
def load_model():
    global model_data
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
    else:
        print("Warning: Model not found. Run train.py first.")

@app.get("/health")
def health_check():
    return {"status": "active", "model_loaded": model_data is not None}

@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    top_movies = model_data.get('top_movies', [])
    user_history = model_data.get('user_history', {}).get(user_id, [])
    recommendations = [m for m in top_movies if m not in user_history][:5]
    
    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "already_watched": user_history
    }
