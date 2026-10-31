# Evaluator-Generator Knowledge Assistant (Cellula AI Engineering Internship)

An intelligent AI-powered question-answering platform built using a **Generator–Evaluator workflow**. The system allows users to provide external knowledge from different sources and uses Retrieval-Augmented Generation (RAG) to generate answers based on the provided information.

The platform uses two AI agents:
- **Generator:** Generates answers based on the retrieved knowledge.
- **Evaluator:** Reviews the generated answers, checks their quality, and provides feedback for improvement.

The Generator and Evaluator work together in an iterative feedback loop to improve answer quality.

## Features

- **Multiple Knowledge Sources:** Supports different types of external knowledge, including PDF, DOCX, TXT, source code, PPT, PPTX, web URLs, Wikipedia pages, and audio files (WAV).
- **RAG (Retrieval-Augmented Generation):** Retrieves relevant information from the knowledge base before generating an answer.
- **Generator Agent:** Generates answers using the retrieved context.
- **Evaluator Agent:** Evaluates answers based on multiple quality criteria.
- **Feedback Loop:** Uses evaluator feedback to improve rejected answers through multiple iterations.
- **Conversation Memory:** Maintains conversation context and previous interactions.
- **Caching with Redis:** Stores frequently used results to reduce repeated operations and improve performance.
- **Vector Database:** Uses ChromaDB to store and retrieve document embeddings.
- **Error Handling and Logging:** Handles errors during the workflow and records important system events.
- **Streamlit Interface:** Provides a user interface for interacting with the platform.

## System Architecture

The system follows an Evaluator–Generator workflow:

1. **Knowledge Ingestion:** The user provides external knowledge through supported file types or other sources.
2. **Text Processing:** The content is extracted, processed, and divided into smaller chunks.
3. **Embedding Generation:** Text chunks are converted into vector embeddings.
4. **Knowledge Storage:** The embeddings and their related content are stored in ChromaDB.
5. **Question Input:** The user submits a question through the application.
6. **Retrieval:** The system searches the knowledge base for relevant information.
7. **Answer Generation:** The Generator creates an answer using the retrieved context.
8. **Answer Evaluation:** The Evaluator checks the answer and provides scores and feedback.
9. **Feedback Loop:** If the answer is not approved, the Generator receives the feedback and attempts to improve the answer.
10. **Final Result:** The system returns the approved answer or the best available answer with an indication that it was not approved.

The workflow supports up to **4 iterations** for generating and evaluating an answer.

## Generator and Evaluator Agents

### Generator Agent

The Generator is responsible for:
- Understanding the user's question.
- Using the retrieved context to generate an answer.
- Following the provided instructions and avoiding unsupported information.
- Improving previous answers based on evaluator feedback.
- Using conversation history when enabled.

### Evaluator Agent

The Evaluator reviews generated answers using the following criteria:

- Accuracy
- Relevance
- Completeness
- Consistency
- Grounding
- No Unsupported Claims
- Overall Quality

It returns an evaluation containing:
- Approval status.
- Scores for the evaluation criteria.
- Unsupported claims.
- Feedback for improving the answer.

## Memory and Caching

The project uses different types of memory and caching for different purposes.

### Generator Memory

Stores previous questions, generated answers, attempts, and retrieved contexts during the application session.

### Evaluator Memory

Stores previous evaluations, scores, and feedback during the application session.

### ChromaDB

Used as the persistent vector database for storing document chunks and their embeddings.

### Redis

Used for caching different types of data, including:
- Final approved answers.
- Retrieval results.
- Generator answers.
- Evaluator decisions.
- Embeddings.
- Audio transcripts.

Redis caching helps reduce repeated operations and improve performance. If Redis is unavailable, the application can continue working without caching.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | User interface |
| LangChain | LLM integration and workflow chains |
| OpenRouter | Access to Large Language Models |
| ChromaDB | Vector database |
| Hugging Face | Text embedding model |
| Sentence Transformers | Text embeddings |
| Redis | Caching |
| Docker | Containerized execution where configured |

## Project Structure

The project is organized into several main components:

- `agents/`: Contains the Generator, Evaluator, LLM configuration, and memory management.
- `workflow/`: Handles the workflow execution, retrieval, iteration loop, and result formatting.
- `knowledge/`: Manages the vector database, embeddings, and Redis caching.
- `ingestion/`: Handles importing and processing external knowledge sources.

Each component has a specific responsibility, making the project easier to maintain and extend.

## Configuration

The project uses environment variables for configuration.

Create a `.env` file in the project root and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Keep your API key private. Do not upload the `.env` file to GitHub.

## Installation and Setup

1. Clone the repository:

```bash
git clone <repository-url>
```

2. Navigate to the project directory:

```bash
cd evaluator-generator-project
```

3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the virtual environment:

**Windows (Git Bash):**

```bash
source .venv/Scripts/activate
```

5. Install the project dependencies:

```bash
pip install -r requirements.txt
```

6. Configure the environment variables in the `.env` file.

7. Make sure Redis is running if you want to use caching.

8. Start the Streamlit application using the project's main application file.

## Caching Behavior

The project uses Redis to cache results and reduce unnecessary processing.

- Standard cache entries are stored for 1 hour.
- Embeddings are cached for 7 days.
- Knowledge versioning helps prevent the use of outdated knowledge-dependent cache entries when the knowledge base changes.

The application can still run when Redis is unavailable, but caching features will not be available.

## Evaluation Workflow

The Evaluator checks each generated answer against the retrieved context.

If an answer is rejected:
1. The Evaluator provides feedback.
2. The feedback is passed to the Generator.
3. The Generator produces an improved answer.
4. The Evaluator reviews the new answer.

This process continues until the answer is approved, an evaluation error occurs, or the maximum number of iterations is reached.

## Future Improvements

- Improve retrieval quality and relevance.
- Support additional knowledge sources.
- Improve evaluation accuracy.
- Enhance conversation memory management.
- Add more detailed performance monitoring.

AI Engineering Project

