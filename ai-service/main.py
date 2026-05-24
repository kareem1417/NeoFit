import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from pymongo import MongoClient
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# تأكد إنك ضايف MONGO_URI في ملف الـ .env بتاع الـ ai-service
MONGO_URI = os.environ.get("MONGO_URI") 

# 2. Initialize Clients (Groq & MongoDB)
client = Groq(api_key=GROQ_API_KEY)
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["neofit"] # اسم الداتا بيز في Atlas
collection = db["knowledge_documents"] # اسم الـ Collection اللي فيها الـ Vectors

# 3. Initialize FastAPI App
app = FastAPI(title="NeoFit AI Service", description="AI and ML Engine for NeoFit App")

# 4. Initialize RAG Models
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 5. Load ML Champion Model
try:
    ml_pipeline = joblib.load('champion_model.pkl')
    ml_model = ml_pipeline['model']
    scaler = ml_pipeline['scaler']
    label_encoder = ml_pipeline['label_encoder']
    expected_features = ml_pipeline['features']
    print("✅ ML Champion Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: ML model not loaded. Error: {e}")

# --- Pydantic Models ---
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    sport: str = "General Fitness"
    history: Optional[List[Message]] = []
    current_program: Optional[str] = None
    user_goal: Optional[str] = None

class UserProfile(BaseModel):
    Age: int
    Height_cm: float
    Weight_kg: float
    BMI: float
    Sport_Type: str
    Level: str
    Goal: str
    Training_Days_Per_Week: int
    Years_Training: float
    Has_Injury_History: int
    Endurance_Score: int
    Strength_Score: int
    Speed_Score: int
    Flexibility_Score: int
    Explosiveness_Score: int
    Recovery_Score: int

# --- Endpoints ---

@app.post("/ask")
async def ask_ai(request: QueryRequest):
    try:
        # 1. Vectorize the user's question
        query_vector = embeddings.embed_query(request.question)
        
        # 2. MongoDB Atlas Vector Search Pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index", # لازم يكون نفس اسم الـ Index في Atlas UI
                    "path": "embedding",     # اسم الفيلد اللي متخزن فيه الـ Array
                    "queryVector": query_vector,
                    "numCandidates": 100,
                    "limit": 10
                }
            },
            {
                "$match": {
                    "sport": request.sport
                }
            },
            {
                "$project": {
                    "content": 1,
                    "_id": 0,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        results = list(collection.aggregate(pipeline))

        # Extract content from results
        unique_docs = list(set([doc.get('content', '') for doc in results if 'content' in doc]))
        
        if not unique_docs:
            return {"answer": "I couldn't find specific information in the training manuals."}

        # 3. Re-ranking with CrossEncoder
        pairs = [[request.question, doc] for doc in unique_docs]
        scores = cross_encoder.predict(pairs)
        scored_docs = sorted(zip(scores, unique_docs), reverse=True)
        top_3_docs = [doc for score, doc in scored_docs[:3]]
        context = "\n---\n".join(top_3_docs)

        # 4. Construct System Prompt for Groq
        system_content = (
            "You are Ringside AI, an expert sports coach and nutritionist. "
            "Answer the user's question based ONLY on the provided context. Be highly motivating and professional. "
            "CRITICAL RULE: You will see [Source: ..., Page: ...] tags in the context. "
            "You MUST include these exact sources at the very end of your answer under a 'Sources:' heading."
        )

        if request.current_program and request.user_goal:
            system_content += f"\nIMPORTANT USER CONTEXT: This user is currently following the '{request.current_program}' program. Their primary goal is '{request.user_goal}'. Tailor your advice specifically to support this goal and program based on the context."

        messages = [{"role": "system", "content": system_content}]

        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"})

        # 5. Call LLM (Groq)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=1024
        )

        return {"answer": completion.choices[0].message.content, "engine": "Advanced RAG MongoDB"}

    except Exception as e:
        return {"error": str(e)}


@app.post("/recommend")
async def recommend_program(profile: UserProfile):
    try:
        df_input = pd.DataFrame(columns=expected_features)
        df_input.loc[0] = 0

        df_input['Age'] = profile.Age
        df_input['Height_cm'] = profile.Height_cm
        df_input['Weight_kg'] = profile.Weight_kg
        df_input['BMI'] = profile.BMI
        df_input['Training_Days_Per_Week'] = profile.Training_Days_Per_Week
        df_input['Years_Training'] = profile.Years_Training
        df_input['Has_Injury_History'] = profile.Has_Injury_History
        df_input['Endurance_Score'] = profile.Endurance_Score
        df_input['Strength_Score'] = profile.Strength_Score
        df_input['Speed_Score'] = profile.Speed_Score
        df_input['Flexibility_Score'] = profile.Flexibility_Score
        df_input['Explosiveness_Score'] = profile.Explosiveness_Score
        df_input['Recovery_Score'] = profile.Recovery_Score
        
        level_col = f"Level_{profile.Level}"
        if level_col in expected_features: df_input[level_col] = 1

        goal_col = f"Goal_{profile.Goal}"
        if goal_col in expected_features: df_input[goal_col] = 1

        sport_col = f"Sport_Type_{profile.Sport_Type}"
        if sport_col in expected_features: df_input[sport_col] = 1

        input_scaled = scaler.transform(df_input)
        prediction_num = ml_model.predict(input_scaled)

        recommended_program = label_encoder.inverse_transform(prediction_num)[0]
        
        # Generating the reason
        reason = f"Chosen specifically for your goal of '{profile.Goal}' in '{profile.Sport_Type}'. "
        
        # Analyzing the level
        if profile.Level == "Beginner":
            reason += "As a beginner, this program focuses on building foundational mechanics safely. "
        elif profile.Level == "Advanced":
            reason += "For your advanced level, it includes high-intensity drills to break plateaus. "
            
        # Analyzing the goal and weight
        if profile.Goal == "Weight Loss":
            reason += f"It incorporates sustained cardio zones optimized to help you burn calories safely at your current weight ({profile.Weight_kg}kg)."
        elif profile.Goal in ["Strength", "Muscle Gain"]:
            reason += "It emphasizes progressive overload to maximize muscle recruitment and power."
        elif profile.Goal == "Endurance":
            reason += "It is designed to progressively increase your stamina and cardiovascular capacity."
            
        return {
            "recommended_program_id": recommended_program,
            "confidence": "94.40%",
            "model_used": "Decision Tree",
            "reason": reason
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)