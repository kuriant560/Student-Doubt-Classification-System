# Student Doubt Classification System

A Streamlit-based academic question answering app that combines document retrieval with local AI fallback.

The system accepts academic content from built-in subject files or uploaded documents, finds the most relevant text using TF-IDF and cosine similarity, and returns a student-friendly answer. If the document match is weak, it falls back to a local Ollama model.

## Features

- Ask academic questions through a simple Streamlit interface
- Use default subject files from the `data/` folder
- Upload `.txt`, `.pdf`, `.docx`, and `.pptx` files as a custom knowledge base
- Retrieve the most relevant content using TF-IDF similarity
- Generate fallback answers with Ollama when document confidence is low
- Show debug information such as similarity score, threshold, and top matches

## Project Structure

```text
Student-Doubt-Classification-System/
├── app.py
├── requirements.txt
├── data/
│   ├── dbms.txt
│   ├── ds.txt
│   ├── ml.txt
│   └── nlp.txt
└── README.md
```

## Tech Stack

- Python
- Streamlit
- scikit-learn
- pdfplumber
- python-docx
- python-pptx
- Ollama

## How It Works

1. The app loads text from the default `data/` files or from uploaded documents.
2. Each document is split into chunks.
3. The user query and document chunks are vectorized using TF-IDF.
4. Cosine similarity is used to find the best matching chunk.
5. If the similarity score is above the threshold, the answer is generated from the matched document content.
6. If the score is below the threshold, the app uses Ollama to generate a fallback answer.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/kuriant560/Student-Doubt-Classification-System.git
cd Student-Doubt-Classification-System
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and run Ollama

Download Ollama from:

`https://ollama.com/download`

Then pull the model used by the app:

```bash
ollama pull llama3.2
```

Make sure Ollama is running locally before using AI fallback.

### 5. Run the app

```bash
streamlit run app.py
```

## Usage

- Launch the app in your browser
- Ask a question in the text box
- Optionally upload academic files in the sidebar
- Review the answer, confidence level, detected subject, and debug info

## Notes

- If Ollama is not running, document retrieval still works, but AI fallback will be unavailable.
- The app is best suited for short academic concept questions.
- Accuracy depends on the quality of the uploaded or default source documents.

## Future Improvements

- Add better query expansion for more subjects
- Improve answer formatting for document-only responses
- Add support for more retrieval strategies
- Add tests and deployment instructions

## License

This project is currently unlicensed. Add a license before wider public distribution if needed.
