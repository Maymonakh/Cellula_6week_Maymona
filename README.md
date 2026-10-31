# Evaluator-Generator Knowledge Assistant (Cellula AI Engineering Internship)

Ask questions about your own documents and web pages. The Generator creates an answer from the retrieved information, and the Evaluator checks whether the answer is supported by the sources.

## How it works

```text
Upload files / add a web page or Wikipedia article
    |
Extract text and split it into chunks
    |
Sentence Transformers creates embeddings
    |
Chroma stores the knowledge
    |
Question -> retrieve relevant chunks
    |
Generator writes an answer
    |
Evaluator checks the answer and gives feedback
    |-- not approved -> Generator tries again
    |
Answer + source information
```

## Project structure

```text
finalsol/
├── app.py                    # Streamlit app
├── requirements.txt
├── docker-compose.yml         # Redis service
├── agents/
│   ├── generator.py           # Generates answers from retrieved context
│   ├── evaluator.py           # Evaluates answers and provides feedback
│   ├── llm.py                 # OpenRouter LLM setup
│   └── memory.py              # Session conversation and agent memory
├── ingestion/
│   ├── loaders.py             # Selects the right file loader
│   ├── documents.py           # Loads documents and code files
│   ├── audio.py               # Transcribes WAV audio with Whisper
│   └── web.py                 # Loads web pages and Wikipedia
├── knowledge/
│   ├── vector_store.py        # Embeddings and Chroma vector database
│   └── cache.py               # Optional Redis cache
└── workflow/
    ├── graph.py               # Generator-Evaluator workflow
    ├── chain.py               # Connects the workflow steps
    ├── retrieval.py           # Retrieves relevant information
    └── results.py             # Formats results
```

## How to run

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your OpenRouter API key in the `OPENROUTER_API_KEY` environment variable.
3. Optional: start Redis for caching:
   ```bash
   docker compose up -d redis
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

Open the local URL shown in the terminal. The first run may take longer while the embedding and Whisper models are downloaded. Redis is optional; the app can run without it.

## Try it

1. Upload a PDF, DOCX, TXT, PowerPoint, code file, or WAV audio file, or add a web page/Wikipedia article.
2. Click the button to process the source.
3. Ask a question about the information you added.
4. Review the answer, its source information, and the Evaluator's feedback.
