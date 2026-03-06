import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
MODEL_PATH = r"C:\Users\hp\Desktop\models\models\all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLLECTION_NAME = "recommender_system"
DIMENSION = 128 

print(f"🚀 Running on: {DEVICE.upper()}")

# Tower Architecture
class TowerMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TowerMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x): return self.net(x)

# Load Models
st_model = SentenceTransformer(MODEL_PATH, device=DEVICE)
item_tower = TowerMLP(387, DIMENSION).to(DEVICE) # 384+3
user_tower = TowerMLP(401, DIMENSION).to(DEVICE) # 384+17

# Load trained weights
item_tower.load_state_dict(torch.load("item_tower_final.pth", map_location=DEVICE))
user_tower.load_state_dict(torch.load("user_tower_final.pth", map_location=DEVICE))
item_tower.eval()
user_tower.eval()

# ==========================================
# 2. MILVUS CONNECTION & COLLECTION SETUP
# ==========================================
connections.connect("default", host="127.0.0.1", port="19530")

if not utility.has_collection(COLLECTION_NAME):
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="product_name", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=200)
    ]
    schema = CollectionSchema(fields, "Two-Tower Aligned Recommendations")
    collection = Collection(COLLECTION_NAME, schema)
    
    # Create HNSW Index
    index_params = {
        "metric_type": "L2", 
        "index_type": "HNSW", 
        "params": {"M": 16, "efConstruction": 128}
    }
    collection.create_index(field_name="embeddings", index_params=index_params)
    print("✅ Collection and HNSW Index created.")
else:
    collection = Collection(COLLECTION_NAME)
    print("✅ Existing collection found.")

# ==========================================
# 3. INSERTION LOGIC (Only if empty)
# ==========================================
def insert_candidates():
    print("⏳ Generating embeddings and inserting candidates...")
    df = pd.read_csv('BigBasket Products.csv')
    
    # 1. PEHLE DATA CLEANING (Most Important for Scaler)
    # Numerical columns se strings aur NaNs hatana
    for col in ['sale_price', 'market_price', 'rating']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Text columns ko fill karna
    df = df.fillna('')

    # 2. SEMANTIC TEXT PREP
    text_content = (df['product'].astype(str) + " " + df['brand'].astype(str) + " " + 
                    df['category'].astype(str) + " " + df['sub_category'].astype(str) + 
                    " " + df['type'].astype(str) + " " + df['description'].astype(str))
    
    base_vecs = st_model.encode(text_content.tolist(), show_progress_bar=True, convert_to_tensor=True)
    
    # 3. NUMERICAL SCALING (Ab error nahi aayega kyunki data float64 mein convert ho chuka hai)
    num_scaler = MinMaxScaler()
    num_vecs = num_scaler.fit_transform(df[['sale_price', 'market_price', 'rating']])
    
    # 4. ALIGNMENT & INSERTION
    with torch.no_grad():
        combined_input = torch.cat([base_vecs, torch.tensor(num_vecs, device=DEVICE)], dim=1).float()
        aligned_vecs = item_tower(combined_input).cpu().numpy()

    data = [
        df['index'].tolist(),
        aligned_vecs.tolist(),
        df['product'].tolist(),
        df['category'].tolist()
    ]
    collection.insert(data)
    collection.flush()
    print(f"✅ Successfully inserted {collection.num_entities} products.")
    
# ==========================================
# 4. RECOMMENDATION LOGIC
# ==========================================
def get_recommendations(user_row, top_k=100):
    # Prepare User Input
    u_text = str(user_row['profile_bio']) + " " + str(user_row['last_search_query'])
    u_base_text = st_model.encode([u_text], convert_to_tensor=True)
    
    aff_cols = [c for c in user_row.index if 'aff_' in c]
    behavior_cols = ['age', 'total_clicks_30d', 'avg_dwell_time', 'search_count_30d', 
                     'avg_ticket_size', 'premium_affinity', 'buying_intent_score']
    
    u_num = user_row[aff_cols + behavior_cols].values.reshape(1, -1).astype(float)
    # Simple normalization for the sample
    u_num = (u_num - 0) / (u_num.max() + 1e-6) 
    u_num_tensor = torch.tensor(u_num, device=DEVICE).float()

    with torch.no_grad():
        u_input = torch.cat([u_base_text, u_num_tensor], dim=1)
        user_embedding = user_tower(u_input).cpu().numpy()

    # Search in Milvus
    collection.load()
    search_params = {"metric_type": "L2", "params": {"ef": 128}}
    results = collection.search(
        data=user_embedding.tolist(), 
        anns_field="embeddings", 
        param=search_params, 
        limit=top_k,
        output_fields=["product_name", "category"]
    )
    return results[0]

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Insert if Milvus is empty
    if collection.num_entities == 0:
        insert_candidates()
    else:
        print(f"ℹ️ Milvus already has {collection.num_entities} items. Skipping insertion.")

    # 2. Load User and Recommend
    print("\n🔍 Fetching recommendations for USR_2000...")
    users_df = pd.read_csv('user_data_50_cols.csv')
    sample_user = users_df.iloc[0] # USR_2000
    
    recs = get_recommendations(sample_user)

    print(f"\n✨ Top 5 Recommendations for {sample_user['user_id']}:")
    for hit in recs:
        p_name = hit.entity.get('product_name')
        cat = hit.entity.get('category')
        print(f"-> {p_name} | [{cat}] | Distance: {hit.distance:.4f}")