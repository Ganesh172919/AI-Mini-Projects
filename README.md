# AI Learning & Experiments

A collection of AI/ML experiments and learning projects, progressing from API-based models to building a custom Large Language Model from scratch.

## Projects

### 1. LangChain Basics (`langchain.py`)
Introduction to LangChain with Google Gemini API:
- Using `ChatGoogleGenerativeAI` for text generation
- Prompt templates and LLM chains
- Vision model integration with Gemini Flash

### 2. AI Tutor with Memory (`learning_ai.py`)
An interactive AI tutor built with LangChain:
- Conversational memory using `RunnableWithMessageHistory`
- Step-by-step topic explanations with real-world examples
- MCQ generation and answer evaluation
- Personalized feedback and scoring

### 3. RAG - PDF Question Answering (`practice.py`)
Retrieval-Augmented Generation pipeline:
- PDF document loading with `PyPDFLoader`
- Vector embeddings with OpenAI
- FAISS vector store for similarity search
- Question answering over documents

### 4. HuggingFace Datasets (`untitled10.py`)
Exploring the HuggingFace datasets library:
- Loading datasets in streaming mode
- Working with the `smolagents/android-control` dataset

### 5. URL Classification with SVM (`url_classify_svm.py`)
Machine learning for cybersecurity:
- URL feature extraction (length, tokens, suspicious keywords)
- SVM classifier for malicious URL detection
- Text preprocessing and cleaning

### 6. 12M Parameter LLM (`llm_12m.py`)
A complete Large Language Model built from scratch:
- Transformer decoder-only architecture (GPT-style)
- ~12 million trainable parameters
- Multi-head self-attention with RoPE
- SwiGLU activation, RMSNorm, GQA support
- Full training loop with learning rate scheduling
- Text generation with temperature, top-k, and top-p sampling

## Tech Stack

- **Frameworks:** PyTorch, LangChain, HuggingFace
- **Models:** Google Gemini, OpenAI GPT
- **ML Libraries:** scikit-learn, FAISS, NumPy, Pandas
- **Environment:** Google Colab, Python

## Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd AI

# Install dependencies
pip install torch numpy

# Run the 12M parameter LLM
python llm_12m.py
```

## License

This project is for educational purposes.
