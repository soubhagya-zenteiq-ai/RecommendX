import torch
import os
from sentence_transformers import SentenceTransformer, util

# 1. Setup Model 
MODEL_PATH = r"C:\Users\hp\Desktop\models\models\all-MiniLM-L6-v2"
# Device check (Industry practice)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_PATH, device=device)

def classify_multi_label(text, candidate_labels, top_k=2):
    """
    Text ko multiple categories mein classify karta hai using Cosine Similarity.
    """
    # Step A: Embeddings generate karein
    text_embedding = model.encode(text, convert_to_tensor=True)
    label_embeddings = model.encode(candidate_labels, convert_to_tensor=True)
    
    # Step B: Cosine Similarity calculate karein
    cosine_scores = util.cos_sim(text_embedding, label_embeddings)[0]
    
    # Step C: Top K results nikalen
    top_results = torch.topk(cosine_scores, k=min(top_k, len(candidate_labels)))
    
    results = []
    print(f"\n✨ Analysis for Post Content:")
    print("-" * 30)
    for i in range(len(top_results.indices)):
        label = candidate_labels[top_results.indices[i]]
        score = top_results.values[i].item()
        results.append((label, score))
        print(f"{i+1}. {label} | Match Score: {score:.4f}")
    
    return results

# --- DATA DEFINITION ---

post_body = """Building production-ready AI systems is not just about writing pip install torch. 
Over the last few months, I've been deep-diving into Natural Language Processing (NLP) and Vector Databases like Milvus.

The real challenge wasn't just training a Two-Tower model; it was handling real-time data and the 'Cold Start' problem for new users. 
Using Python and PyTorch, I managed to align user preferences with product semantics in a shared 128-dimensional embedding space.

For my fellow 2026 batch mates: The job market is shifting towards Agentic AI. 
Focus on building end-to-end pipelines rather than just following Kaggle tutorials. 
Consistency is key! If you are looking for internships in AI or Data Science, make sure your portfolio reflects 
real-world problem-solving, not just academic projects.

#AI #MachineLearning #CareerAdvice #Python #Batch2026 #TechCommunity"""

# Labels jo aapke user affinity columns se match karte hain
labels = ["Technology", "Career Advice", "Fitness", "Business", "Gaming", "Hiring"]

# --- EXECUTION ---

# top_k=2 kyunki ek post Technical bhi ho sakti hai aur Career-oriented bhi
detected_categories = classify_multi_label(post_body, labels, top_k=2)

# Global variables print karne ke liye (Aapke original query ke fix ke liye)
top_category, confidence = detected_categories[0]
print(f"\n🎯 Final Verdict: This post is primarily about '{top_category}'")