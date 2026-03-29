<div align="center">
<h1>Semantic Graph-RAG: Enhancing Scientific Literature Retrieval via <br>Citation-Aware Graph Attention Networks</h1>

**A Research Project by [Your Name]**
<br>
*Optimizing Pre-trained Sentence Embeddings for Topology-Aware Semantic Search*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![GAT](https://img.shields.io/badge/GAT-Graph_Attention-green.svg)](https://pytorch-geometric.readthedocs.io/)
[![Sentence-Transformers](https://img.shields.io/badge/Embeddings-Sentence--Transformers-orange.svg)](https://huggingface.co/sentence-transformers)
[![PyVis](https://img.shields.io/badge/Visualization-PyVis-purple.svg)](https://pyvis.readthedocs.io/)

</div>

---

## 📌 Project Overview

**Semantic Graph-RAG** is an advanced Retrieval-Augmented Generation (RAG) framework designed to bridge the gap between pure semantic text mapping and latent academic relationships. It introduces **Graph Attention Networks (GAT)** to model the intrinsic citation and conceptual topology between scientific papers, fundamentally upgrading how Large Language Models (LLMs) retrieve contextual information.

### 🔍 The Problem
Traditional Semantic Search Engines (like base Sentence-BERT) often suffer from the **Semantic Gap** when applied to scientific literature due to:
1.  **Contextual Blindness**: Treating documents as isolated text chunks ignores the rich, interconnected ecosystem of academic citations.
2.  **Vocabulary Mismatch**: Papers discussing the exact same underlying concepts might use vastly different terminologies, avoiding traditional vector collision.

### 💡 The GAT & Graph-RAG Solution
By integrating a **GAT layer**, this project enables the retrieval model to perform "cross-document reasoning." Instead of treating papers as isolated vectors, the GAT refines feature embeddings by attending to citation neighbors (KNN Graph). 
Furthermore, the **Graph-RAG** module dynamically expands the retrieved context by pulling in related explicit concepts from a constructed Knowledge Graph, providing LLMs with a mathematically enriched prompt for highly accurate generation.

---

## 🛠️ Tech Stack & Skills
- **Deep Learning Frameworks**: PyTorch, PyTorch Geometric (GAT implementation).
- **NLP & Embeddings**: HuggingFace `sentence-transformers` (all-MiniLM-L6-v2).
- **Graph Engineering**: NetworkX (Knowledge Graph construction), PyVis (Interactive 3D Barnes-Hut Physics Visualization).
- **Methodologies**: Retrieval-Augmented Generation (RAG), Geometric Deep Learning (GDL), Semantic Search, Link Prediction Task.
- **Infrastructure**: Automated Multi-stage Pipelines, Pandas Data Processing.

---

## 🚀 Execution Guide

The system operates on a highly modular, 3-Pillar pipeline. Ensure your environment is configured per `env/environment.yml` before executing.

### Pillar 1: Knowledge Graph Construction
Extracts entities and conceptual metadata to build the foundational Multi-relational Graph.
```bash
python kg_builder.py
```
*(Optional)* Generate an interactive 3D HTML visualization of the Knowledge Graph:
```bash
python visualize_kg.py
```

### Pillar 2: GAT Representation Learning
Mines raw representations and finetunes them using Citation Topology.
```bash
# Extract base sentence embeddings
python scripts/embed_papers_hf.py --data-path papers_clean.json --output raw_embeddings.jsonl

# Train GAT to inject citation genetics into the embeddings
python scripts/gat_refinement.py --embeddings raw_embeddings.jsonl --metadata papers_clean.json --output refined_embeddings.jsonl --epochs 100
```

### Pillar 3: Graph-RAG Hybrid Search Engine
Injects the GAT-refined vectors into the database and runs the expanded semantic search.
```bash
# Assimilate GAT vectors into the central CSV database
python database_vector.py

# Execute the Hybrid Vector Dense Search + KG Context Expansion
python search_module.py
```

---

## 📁 Repository Structure
- `scripts/embed_papers_hf.py`: Generates base dense embeddings via `Sentence-Transformers` (bypassing PyTorch CVE-2025-32434 restrictions).
- `scripts/gat_refinement.py`: The core Geometric Deep Learning engine optimizing node features via Graph Attention.
- `kg_builder.py` & `visualize_kg.py`: Automated NLP entity extraction and 3D web-based visualization.
- `search_module.py` & `database_vector.py`: The Retrieval-Augmented Generation context assembler.

---
*Created as part of a University Research Thesis. For technical collaboration, please contact me via GitHub or Email.*
