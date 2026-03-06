import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import os

# ==========================================
# 1. CONFIGURATIONS & PATHS
# ==========================================
# Path update karein agar alag hai
MODEL_PATH = r"C:\Users\hp\Desktop\models\models\all-MiniLM-L6-v2" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECTION_DIM = 128  # Aligned vector size
BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 0.0005

print(f"🚀 Running on: {DEVICE.upper()}")

# ==========================================
# 2. FEATURE ENGINEERING (The "Must" Columns)
# ==========================================
def get_item_base_vectors(df, st_model):
    print("📦 Processing Item Features (10 Columns)...")
    df = df.fillna({'description': '', 'brand': 'Unknown', 'rating': 0.0})
    
    # Textual: Semantic combination
    text_content = (df['product'] + " " + df['brand'] + " " + df['category'] + 
                    " " + df['sub_category'] + " " + df['type'] + " " + df['description'])
    text_vecs = st_model.encode(text_content.tolist(), show_progress_bar=True)
    
    # Numerical: Price & Rating
    num_cols = ['sale_price', 'market_price', 'rating']
    scaler = MinMaxScaler()
    num_vecs = scaler.fit_transform(df[num_cols])
    
    return np.hstack([text_vecs, num_vecs]) # 384 + 3 = 387 dim

def get_user_base_vectors(df, st_model):
    print("👤 Processing User Features (Behavioral + Affinities)...")
    # Textual: Bio & Intent
    user_text = df['profile_bio'] + " " + df['last_search_query']
    text_vecs = st_model.encode(user_text.tolist(), show_progress_bar=True)
    
    # Numerical: Affinities (10) + Behavioral (7)
    aff_cols = [c for c in df.columns if 'aff_' in c]
    behavior_cols = ['age', 'total_clicks_30d', 'avg_dwell_time', 'search_count_30d', 
                     'avg_ticket_size', 'premium_affinity', 'buying_intent_score']
    
    scaler = MinMaxScaler()
    num_vecs = scaler.fit_transform(df[aff_cols + behavior_cols])
    
    return np.hstack([text_vecs, num_vecs]) # 384 + 17 = 401 dim

# ==========================================
# 3. MODEL ARCHITECTURE (Best Practices)
# ==========================================
class TowerMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TowerMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512), # Best Practice: Training stable rakhta hai
            nn.ReLU(),
            nn.Dropout(0.3),     # Best Practice: Overfitting rokta hai
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. TRAINING WORKFLOW
# ==========================================
if __name__ == "__main__":
    # Load Data
    p_df = pd.read_csv('BigBasket Products.csv')
    u_df = pd.read_csv('user_data_50_cols.csv')
    i_df = pd.read_csv('interactions.csv')

    # Initialize MiniLM
    st_model = SentenceTransformer(MODEL_PATH, device=DEVICE)

    # Prepare Vectors
    item_base = get_item_base_vectors(p_df, st_model)
    user_base = get_user_base_vectors(u_df, st_model)
    
    # Create Lookups
    item_map = {row['index']: item_base[i] for i, row in p_df.iterrows()}
    user_map = {row['user_id']: user_base[i] for i, row in u_df.iterrows()}

    # Dataset & Loader
    class RecDataset(Dataset):
        def __init__(self, interactions, u_map, i_map):
            self.data = interactions
            self.u_map = u_map
            self.i_map = i_map
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            return torch.tensor(self.u_map[row['user_id']], dtype=torch.float32), \
                   torch.tensor(self.i_map[row['product_index']], dtype=torch.float32), \
                   torch.tensor(row['label'], dtype=torch.float32)

    train_loader = DataLoader(RecDataset(i_df, user_map, item_map), batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Towers
    user_tower = TowerMLP(user_base.shape[1], PROJECTION_DIM).to(DEVICE)
    item_tower = TowerMLP(item_base.shape[1], PROJECTION_DIM).to(DEVICE)

    # Optimizer & Loss
    optimizer = optim.Adam(list(user_tower.parameters()) + list(item_tower.parameters()), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    print("开始 (Start) Training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for u_batch, i_batch, l_batch in train_loader:
            u_batch, i_batch, l_batch = u_batch.to(DEVICE), i_batch.to(DEVICE), l_batch.to(DEVICE)
            
            optimizer.zero_grad()
            u_emb = user_tower(u_batch)
            i_emb = item_tower(i_batch)
            
            # Similarity Calculation (Dot Product)
            logits = torch.sum(u_emb * i_emb, dim=1)
            loss = criterion(logits, l_batch)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    # SAVE RESULTS
    torch.save(user_tower.state_dict(), "user_tower_final.pth")
    torch.save(item_tower.state_dict(), "item_tower_final.pth")
    print("✅ Training Finished. Models Saved!")