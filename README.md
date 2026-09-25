# Domain-Specific RAG Chatbot for PDF Question Answering

## 1. Project Description

The Domain-Specific RAG Chatbot is a question-answering application that allows users to upload PDF documents and ask questions about their content.

The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the uploaded documents and generate answers based only on the retrieved document context.

The chatbot also displays the source document and page number used for the answer.

---

## 2. Problem Statement

Large PDF documents can be difficult and time-consuming to search manually. Users may need to read many pages to find specific information.

This project provides a chatbot interface where users can upload PDF documents and ask questions in natural language. The system searches the uploaded documents and provides relevant answers.

---

## 3. Objectives

- Upload one or multiple PDF documents.
- Extract text from PDF pages.
- Preserve document name and page number as metadata.
- Split large text into smaller chunks.
- Generate embeddings for document chunks.
- Store embeddings using FAISS.
- Retrieve the most relevant document chunks.
- Generate answers using a language model.
- Display the source document and page number.
- Avoid generating answers when the required information is not available.

---

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Streamlit | User interface |
| PyPDF | PDF text extraction |
| LangChain | Text splitting and RAG support |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| FAISS | Vector similarity search |
| Gemini | Language model |
| python-dotenv | Environment variable management |
| Git & GitHub | Version control |

---

## 5. RAG Workflow

The application follows this workflow:

```text
PDF Upload
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Generate Embeddings
     ↓
Store Embeddings in FAISS
     ↓
User Asks a Question
     ↓
Convert Question into Embedding
     ↓
Similarity Search
     ↓
Retrieve Relevant Chunks
     ↓
Send Context + Question to Gemini
     ↓
Generate Grounded Answer
     ↓
Display Answer + Source + Page Number

---

## 6. Main Project Modules

### Module 1: Document Upload

The application allows users to upload one or multiple PDF documents.

The uploaded documents are validated and displayed in the Streamlit interface.

### Module 2: Text Extraction

The application uses PyPDF to read text from each PDF page.

The document name and page number are preserved as metadata.

### Module 3: Text Chunking

Large documents are divided into smaller text chunks.

Chunk overlap is used to preserve connected information between chunks.

### Module 4: Embeddings and Vector Store

Each text chunk is converted into a numerical embedding using the Sentence Transformers model.

The embeddings are stored and searched using FAISS.

### Module 5: Retrieval

When the user asks a question, the question is converted into an embedding.

The system searches FAISS and retrieves the most relevant document chunks.

### Module 6: Answer Generation

The retrieved document context and user question are sent to the Gemini language model.

The chatbot is instructed to answer only using the supplied document context.

If the required information is not available, the chatbot provides a fallback response instead of inventing information.

### Module 7: Streamlit Interface

The Streamlit interface provides:

- PDF upload
- Document processing
- Chat input
- Chat history
- Source information
- Page numbers
- Clear chat functionality

---

## 7. Project Structure

```text
project-3/
│
├── app.py
├── rag_pipeline.py
├── document_loader.py
├── vector_store.py
├── prompt.py
├── requirements.txt
├── README.md
├── architecture.png
├── PROJECT_REPORT.pdf
├── .gitignore
│
├── documents/
│   └── Sample PDF documents
│
└── tests/
    ├── evaluate.py
    ├── test_questions.csv
    └── results.csv

---

## 8. Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>