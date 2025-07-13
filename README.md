# Vector Database Comparison

A comprehensive comparison of popular vector databases and similarity search libraries for semantic search applications. This project evaluates the performance, ease of use, and features of different vector database solutions using real-world news data.

##  Project Overview

This project compares five popular vector database solutions:
- **FAISS** (Facebook AI Similarity Search)
- **Annoy** (Spotify's Approximate Nearest Neighbors)
- **HNSWlib** (Hierarchical Navigable Small World)
- **Qdrant** (Vector Database)
- **ChromaDB** (Embedding Database)

The comparison focuses on:
- Indexing time performance
- Search query speed
- Memory usage patterns
- Recall accuracy
- Ease of implementation

##  Dataset

The project uses the AG News dataset, specifically 1,000 news articles for testing. The dataset is loaded using Hugging Face's `datasets` library and processed using the `all-MiniLM-L6-v2` sentence transformer model for generating embeddings.

##  Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Notebook or Google Colab
- Internet connection for downloading packages and data

### Installation

Install the required packages:

```bash
pip install faiss-cpu
pip install chromadb
pip install qdrant-client
pip install sentence-transformers
pip install datasets
pip install annoy
pip install hnswlib
pip install pandas
pip install matplotlib
pip install seaborn
```

Or install all at once:

```bash
pip install faiss-cpu chromadb qdrant-client sentence-transformers datasets annoy hnswlib pandas matplotlib seaborn
```

### Running the Comparison

1. Open the `Vector_DB_Comparison.ipynb` notebook
2. Run all cells sequentially
3. View the performance comparison results and visualizations

## 📈 Performance Results

Based on the comparison with 1,000 news articles:

| Database | Indexing Time (s) | Search Time (s) | Memory Usage | Recall@10 |
|----------|------------------|-----------------|--------------|-----------|
| FAISS    | 0.0223          | 0.0043         | In-Memory    | 0.0       |
| Annoy    | 0.0139          | 0.0001         | In-Memory    | 0.0       |
| HNSWlib  | 0.0582          | 0.0002         | In-Memory    | 0.0       |
| Qdrant   | 0.5643          | 0.0102         | Cloud/In-Memory | 0.0   |
| ChromaDB | 25.3807         | 0.0224         | Disk + Memory | 0.3    |

### Key Findings

- **Fastest Indexing**: Annoy (0.0139s)
- **Fastest Search**: Annoy (0.0001s)
- **Best Recall**: ChromaDB (0.3)
- **Most Memory Efficient**: FAISS, Annoy, HNSWlib (In-Memory)
- **Most Feature-Rich**: ChromaDB (persistent storage, metadata)

## 🔧 Implementation Details

### 1. FAISS (Facebook AI Similarity Search)
```python
import faiss
import numpy as np

# Create index
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Search
D, I = index.search(np.array([query]), k=10)
```

**Pros**: Fast, memory-efficient, production-ready
**Cons**: Basic functionality, limited metadata support

### 2. Annoy (Spotify's Approximate Nearest Neighbors)
```python
from annoy import AnnoyIndex

# Create index
annoy_index = AnnoyIndex(dimension, 'angular')
for i, emb in enumerate(embeddings):
    annoy_index.add_item(i, emb.tolist())
annoy_index.build(10)

# Search
indices = annoy_index.get_nns_by_vector(query.tolist(), 10)
```

**Pros**: Extremely fast search, simple API
**Cons**: Approximate results, limited features

### 3. HNSWlib (Hierarchical Navigable Small World)
```python
import hnswlib

# Create index
hnsw_index = hnswlib.Index(space='cosine', dim=dimension)
hnsw_index.init_index(max_elements=1000, ef_construction=100, M=16)
hnsw_index.add_items(embeddings)

# Search
labels, distances = hnsw_index.knn_query(query, k=10)
```

**Pros**: Good balance of speed and accuracy
**Cons**: Memory-only, no persistence

### 4. Qdrant (Vector Database)
```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Create client and collection
client = QdrantClient(":memory:")
client.recreate_collection(
    collection_name="test_collection",
    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
)

# Add points
points = [PointStruct(id=i, vector=vec, payload={"text": texts[i]}) 
          for i, vec in enumerate(embeddings)]
client.upsert(collection_name="test_collection", points=points)

# Search
hits = client.search(
    collection_name="test_collection",
    query_vector=query.tolist(),
    limit=10
)
```

**Pros**: Rich metadata support, cloud deployment, filtering
**Cons**: Slower indexing, more complex setup

### 5. ChromaDB (Embedding Database)
```python
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Create client and collection
chromadb_client = chromadb.Client()
chroma_embeddings = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chromadb_client.create_collection(name="news", embedding_function=chroma_embeddings)

# Add documents
for i, text in enumerate(texts):
    collection.add(documents=[text], ids=[str(i)])

# Search
results = collection.query(query_texts=[query], n_results=10)
```

**Pros**: Persistent storage, rich metadata, easy to use
**Cons**: Slowest indexing, higher memory usage

##  Visualizations

The project generates three key visualizations:
1. **Indexing Time Comparison** - Shows how long each database takes to build its index
2. **Search Time Comparison** - Shows query response times
3. **Recall@10 Comparison** - Shows accuracy of search results

##  Use Cases and Recommendations

### Choose FAISS when:
- You need maximum speed and memory efficiency
- Working with large-scale datasets
- Building production systems with custom requirements

### Choose Annoy when:
- You need extremely fast approximate search
- Working with smaller datasets
- Simple similarity search is sufficient

### Choose HNSWlib when:
- You need a good balance of speed and accuracy
- Working with medium-sized datasets
- Memory-only storage is acceptable

### Choose Qdrant when:
- You need rich metadata and filtering capabilities
- Planning to deploy to cloud infrastructure
- Building complex search applications

### Choose ChromaDB when:
- You need persistent storage and easy setup
- Working on research or prototyping
- Need rich metadata and document management
