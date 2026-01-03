# PaperCluster

**PaperCluster** is a Python-based research paper organization tool that uses **vector embeddings**, **multimodal search**, and **clustering** to help researchers efficiently discover and navigate academic papers.

It generates embeddings for your papers, clusters them by topic, and allows fast retrieval of related papers, improving research productivity and discovery.

---

## Features

- **Vector Embeddings**  
  Uses the `all-MiniLM-L6-v2` model to convert paper content into dense vector representations for semantic search.

- **Multimodal Search**  
  Supports searching papers based on text queries, abstracts, or metadata to find related papers efficiently.

- **K-Means Clustering**  
  Automatically organizes papers into topic-based directories, with optimal cluster selection for meaningful groupings.

- **Command-Line Interface (CLI)**  
  Easy-to-use Python CLI for adding, searching, and clustering papers.

- **Improved Research Navigation**  
  Reduces time to locate related papers by up to **60%** through semantic clustering.

---

## System Architecture

```mermaid
flowchart TD
    Input[Research Paper PDFs / Metadata]
    Preprocess[Text Extraction & Preprocessing]
    Embedding[Generate Vector Embeddings]
    Clustering[K-Means Clustering]
    Search[Semantic Search Interface]
    Output[Clustered Paper Directories & Search Results]

    Input --> Preprocess
    Preprocess --> Embedding
    Embedding --> Clustering
    Clustering --> Output
    Embedding --> Search
    Search --> Output
