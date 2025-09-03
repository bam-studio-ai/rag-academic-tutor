# RAG-Based Academic Tutor

A comprehensive Retrieval-Augmented Generation (RAG) system for academic document Q&A, featuring multiple chunking strategies, vector embeddings, and Mistral-7B integration with a Streamlit interface.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │───▶│   Processing    │───▶│   Vector        │
│   Ingestion     │    │   & Chunking    │    │   Storage       │
│                 │    │                 │    │   (ChromaDB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │◀───│   RAG Chain     │◀───│   Retrieval     │
│                 │    │   Orchestrator  │    │   Engine        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                       ┌─────────────────┐
                       │   Mistral-7B    │
                       │   Generation    │
                       └─────────────────┘
```

## ✨ Features

### **Document Processing**
- **Multi-format support**: Text files (.txt)
- **Advanced chunking**: Sliding window, paragraph boundary, semantic chunking
- **Text preprocessing**: Whitespace normalization, encoding fixes, citation cleanup

### **Retrieval System**
- **Vector embeddings**: Sentence-transformers integration
- **Vector database**: ChromaDB with persistent storage
- **Similarity search**: Configurable top-k retrieval

### **Language Model**
- **Mistral-7B-Instruct-v0.3**: State-of-the-art instruction-tuned model
- **Mock model support**: For testing and development
- **Optimized inference**: Float16 precision, device mapping

### **User Interface**
- **Streamlit web app**: Interactive chat interface
- **Real-time processing**: Stream responses
- **Session management**: Persistent chat history

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for Mistral-7B)
- 8GB+ GPU memory (16GB+ recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rag-academic-tutor
   cd rag-academic-tutor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

## 💡 Usage

### **Basic Q&A**
1. Launch the Streamlit app: `streamlit run main.py`
2. Upload academic documents via the interface
3. Ask questions about your documents
4. Get contextual answers with source references

### **Programmatic Usage**
```python
from src.generation.rag_chain import RAGChain

# Initialize RAG system
rag = RAGChain(
    persist_directory="./data/chromadb",
    collection_name="my_docs"
)

# Ask questions
response = rag.ask("What is the main hypothesis of this paper?")
print(response.answer)
print(f"Sources: {len(response.source_documents)} documents")
```

## ⚙️ Configuration

### **Model Options**
- **Development**: Use `MockModel` for testing
- **Production**: Use `mistralai/Mistral-7B-Instruct-v0.3`

### **Chunking Strategies**
- `sliding_window`: Fixed-size chunks with overlap
- `paragraph_boundary`: Respects paragraph structure  
- `semantic`: Topic-aware chunking

### **Testing**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest test/ingestion/test_document.py -v
```

## 📁 Project Structure

```
rag-academic-tutor/
├── main.py                 # Streamlit application entry point
├── requirements.txt        # Python dependencies
├── pytest.ini            # Test configuration
├── src/
│   ├── ingestion/         # Document processing & chunking
│   │   ├── document.py    # Document loading
│   │   ├── chunker.py     # Text chunking strategies
│   │   ├── embedding.py   # Embedding models
│   │   └── preprocessor.py # Text cleaning
│   ├── retrieval/         # Vector search & retrieval
│   │   └── vector_store.py
│   └── generation/        # LLM integration & response generation
│       ├── rag_chain.py   # RAG orchestrator
│       ├── llm_client.py  # Mistral-7B client
│       └── prompt_template.py
├── test/                  # Test suite
├── data/                  # Data storage
│   └── chromadb/         # Vector database
└── docs/                  # Documentation
```

## 🔧 Technical Implementation

### **Key Technologies**
- **Language Model**: Mistral-7B-Instruct-v0.3
- **Embeddings**: Sentence-transformers
- **Vector Database**: ChromaDB  
- **Web Framework**: Streamlit
- **ML Framework**: PyTorch + Transformers

### **Performance Optimizations**
- Float16 precision for memory efficiency
- Automatic device mapping for GPU utilization
- Persistent vector storage with ChromaDB
- Configurable chunking for optimal context

## AI Tools Disclaimer

During the development of this project we are limiting the use of AI tools 
to research only. The code in this repository has been organically grown. 
