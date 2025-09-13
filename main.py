import os
import pandas as pd
import shutil
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from kneed import KneeLocator  # pip install kneed

# ------------------- Configuration -------------------
pdf_dir = "papers/"                 # Directory containing PDFs
output_csv = "paper_embeddings.csv" # CSV to save/load embeddings
clustered_dir = "clustered_papers"  # Directory to save clusters
model_name = "all-MiniLM-L6-v2"     # Sentence Transformer model
max_k = 10                           # Maximum clusters to test

# ------------------- Load Model -------------------
model = SentenceTransformer(model_name)

# ------------------- Generate or Load Embeddings -------------------
def gen_embeddings():
    if os.path.exists(output_csv):
        print("CSV already exists, loading embeddings.")
        df = pd.read_csv(output_csv)
        df["embedding"] = df["embedding"].apply(lambda x: np.array(eval(x)))
        return df
    else:
        data = []
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, filename)
                reader = PdfReader(pdf_path)
                text_pages = []
                found_intro = False

                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    if "Introduction" in page_text:
                        found_intro = True
                    if not found_intro:
                        text_pages.append(page_text)

                full_text = "\n".join(text_pages).strip()
                if full_text:
                    embedding = model.encode(full_text)
                    data.append({
                        "filename": filename,
                        "text": full_text,
                        "embedding": embedding.tolist()
                    })

        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(df.head())
        print("Total PDFs processed:", len(df))
        df["embedding"] = df["embedding"].apply(lambda x: np.array(x))
        return df

# ------------------- Automatically find optimal k -------------------
def find_optimal_k(embeddings, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        wcss.append(kmeans.inertia_)

    # Use KneeLocator to find elbow
    kl = KneeLocator(range(1, max_k + 1), wcss, curve="convex", direction="decreasing")
    optimal_k = kl.elbow
    print(f"Optimal number of clusters detected: {optimal_k}")
    return optimal_k

# ------------------- Cluster PDFs -------------------
def cluster_pdfs(df, k):
    embeddings = np.stack(df["embedding"].to_numpy())
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    df["cluster"] = labels

    # Create cluster directories
    os.makedirs(clustered_dir, exist_ok=True)
    for i in range(k):
        os.makedirs(os.path.join(clustered_dir, f"Cluster_{i+1}"), exist_ok=True)

    # Move PDFs into cluster folders
    for idx, row in df.iterrows():
        src_path = os.path.join(pdf_dir, row["filename"])
        dst_path = os.path.join(clustered_dir, f"Cluster_{row['cluster']+1}", row["filename"])
        shutil.copy(src_path, dst_path)

    print(f"PDFs have been clustered into {k} directories inside '{clustered_dir}'.")

# ------------------- Main -------------------
df = gen_embeddings()
embeddings_matrix = np.stack(df["embedding"].to_numpy())
optimal_k = find_optimal_k(embeddings_matrix, max_k=max_k)
cluster_pdfs(df, optimal_k)
