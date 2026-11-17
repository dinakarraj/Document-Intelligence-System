# Document Intelligence System

Ask questions about a research paper and get answers that come only from that paper, with the page numbers they were found on.

I built this to learn how retrieval-augmented generation (RAG) works from the ground up. The first version works on one document: the paper *Attention Is All You Need*. A multi-document version is next.

## What it does

- You type a question in plain English.
- The app finds the most relevant passages in the paper.
- A language model writes the answer using only those passages.
- You see the answer, the pages it came from, and a history of your earlier questions in the sidebar.
- If the paper doesn't cover the question (for example, "Who won the 2018 FIFA World Cup?"), the app says so instead of guessing.

## How it works

1. **Reading the paper.** The PDF is read page by page, cleaned, and split into small chunks of about 400 characters. Each chunk remembers which page it came from.
2. **Storing it.** Each chunk is turned into a list of numbers (an embedding) that captures its meaning, and saved in a local ChromaDB database. This happens once. After that, the database is reused.
3. **Searching.** Your question is turned into numbers the same way, and the 15 closest chunks are pulled out.
4. **Re-ranking.** A second, more careful model reads the question together with each of those 15 chunks and sorts them by how well they actually answer it.
5. **Checking the paper covers it.** If even the best chunk scores badly, the app stops and says the question isn't covered. No answer is generated.
6. **Adding context.** Answers are sometimes split across neighbouring chunks, so the app also pulls in the chunks just before and after the best matches, and keeps them in reading order.
7. **Answering.** The language model gets that text and is told to answer only from it, and to say so when the text isn't enough.

## A problem I ran into

Questions like "What are the three ways multi-head attention is used?" came back with only one of the three. The answer was split across chunk boundaries, and only the first piece was being retrieved. Retrieving more chunks didn't help, because the missing piece wasn't close to being retrieved at all. Giving the model the neighbouring chunks of the best matches fixed it. The same fix completed the answers about the encoder's and decoder's structure.

## Tech used

- Python, Streamlit
- ChromaDB (vector database)
- Gemini embeddings (`gemini-embedding-001`) and a Gemini model for the answers
- A cross-encoder re-ranker from `sentence-transformers` (`ms-marco-MiniLM-L-6-v2`)
- PyMuPDF for reading the PDF

## Run it yourself

1. Clone the repo and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```
2. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
3. Copy `.env.example` to `.env` and put your key in it.
4. Download the paper from [arXiv](https://arxiv.org/abs/1706.03762) and make sure you change file path according to your system path
5. Start the app:
   ```
   python -m streamlit run app.py
   ```
   The first start takes a while because the database is built and the models are loaded.

## Known limits

- It works on one document, with text that can be selected in the PDF. Scanned documents are not supported.
- The sources line lists every page of every chunk sent to the model, not only the ones the answer actually used.
- I checked the answers by hand on 13 test questions. There is no automatic evaluation yet.
- Free-tier API limits can stop the app for the day if you ask many questions.

## What's next

- Upload and search across several documents
- A measured evaluation of how often the right page is found