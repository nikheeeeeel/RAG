# Basic RAG Implementation

I implemented a very generic RAG project, referring to freeCodeCamp. Used LangChain primarily.

## Architecture

**LOAD PDF → SPLIT TEXTS (CHUNKS) → EMBEDDING AND STORE IN VECTOR DB → RETRIEVE K CHUNKS → PROMPT TEMPLATE WITH RETRIEVED INFO → INVOKE LLM WITH RETRIEVED CONTEXT**

### 1. Loading
Using `PyPDFLoader` to load the PDF stored in the root of the folder.

### 2. Chunking
Splitting texts using `RecursiveCharacterTextSplitter`. Here, the `chunk_size` and `chunk_overlap` are mentioned.

### 3. Embedding
`embeddinggemma` is used as the local embedding model to convert the chunks into vectors and store them in the vector DB stored in ChromaDB.

### 4. Retrieval
`k` chunks are set to be retrieved.

### 5. Prompt Template
A prompt template is made which has two placeholders for context and question. This is passed to the LLM.

### 6. Helper Function
The helper function `format_docs()` is used to remove unnecessary info from the chunks and only pass textual info to the LLM.

### 7. RAG Chain

- `context`: `retriever | format_docs`
- `question`: Passed as it is.
- `| prompt`
- `| llm`

Here, `|` works as passing the output of the previous step to the next step.

**[context: retrieved info → formatted, question] → prompt → llm**

### 8. Response
The response is invoked with:

```python
rag_chain.invoke(question)
```