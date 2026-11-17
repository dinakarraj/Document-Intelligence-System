import os, re, pymupdf , time
from google import genai
from google.genai import types, errors
import chromadb 
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

FILE_PATH = r"D:\RAG\Attention_Research_paper.pdf"
LLM_MODEL = "gemini-3.6-flash"
EMBED_MODEL = "gemini-embedding-001"
COLLECTION_NAME = "Attention_Paper_Collection"
SEPARATORS = ["\n\n", "\n", ". ", " "]
CHUNK_SIZE = 400

genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))  


def clean_text(text):
    text = re.sub(r"\n\d+\s*$", "", text)
    return text

def expand_with_neighbors(anchors, window, total):

    result = set()

    for anchor in anchors:
        for n in range(anchor - window, anchor + window + 1):
            if 1 <= n <= total:
                result.add(n)

    return sorted(result)

def generate_answer_LLM(context, query):
    prompt = f"""
      You are Given with a query and context , 
      you have to answer query using only context, 
      dont use any external knowledge part from context, 
      you are given with top n number of chunks (number can be varied), 
      you have to select the relevant and answer it using the only context given to you

       context : {context}
       query   : {query}

        If the context does not contain enough information to answer, say so explicitly rather than guessing
        return the response with out any json markdown or fences.
        """

    for attempt in range(3):
        try:
            response = genai_client.models.generate_content(model=LLM_MODEL, contents = prompt)
            return response.text
        except errors.ServerError:
            if attempt==2:
                raise 
            wait = 2 ** (attempt+1)
            print(f"\nServer busy, retrying in {wait}s...\n")
            time.sleep(wait)
       

def recursive_chunking(text, seperators, max_chunk_size=400):
    if not seperators:
        final_chunks=[]
        for i in range(0, len(text), max_chunk_size):
            final_chunks.append(text[i:i+max_chunk_size])
        return final_chunks

    if len(text)<=max_chunk_size:
        return [text]

    current_seperator = seperators[0]
    remaining_seperators = seperators[1:]
    final_chunks=[]

    pieces = text.split(current_seperator)

    for piece in pieces:
        if len(piece)<=max_chunk_size:
            final_chunks.append(piece)
        else:
            sub_chunks = recursive_chunking(piece, remaining_seperators, max_chunk_size)
            final_chunks.extend(sub_chunks)
    return final_chunks

def merge_chunks(chunks, max_chunk_size=400, glue=" "):

    current_chunk=""
    merged_chunks=[]

    for chunk in chunks:
        if len(chunk)+len(current_chunk)<=max_chunk_size:
            current_chunk += chunk
            current_chunk += glue
        else:
            merged_chunks.append(current_chunk)
            current_chunk=""
            current_chunk += chunk
            current_chunk += glue

    if current_chunk:
        merged_chunks.append(current_chunk)

    return merged_chunks

def setup():

    chroma_client = chromadb.PersistentClient(path="./attention_mechanism")
    
    my_collection = chroma_client.get_or_create_collection(name="Attention_Paper_Collection", 
                                                       embedding_function=embedding_functions.GoogleGeminiEmbeddingFunction(model_name="gemini-embedding-001", 
                                                                                                                            task_type="RETRIEVAL_DOCUMENT", 
                                                                                                                            api_key_env_var="GEMINI_API_KEY")
                                                       )
    # you can choose any one of multiple re rankers

    re_ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return my_collection, re_ranker

def ingest(file_path, collection):

    if collection.count()>0:
        print(f"\nThe document was already ingested and it has {collection.count()} chunks, skipping ingesion\n")
        return 

    
    document = pymupdf.open(file_path)
    all_documents = []
    all_meta_datas=[]
    all_ids=[]

    seperators=["\n\n", "\n", ". ", " "]
    id_counter = 0

    for i in range(len(document)):
        page_content = document[i].get_text()
        cleaned_page_content=clean_text(page_content)

        chunks = recursive_chunking(cleaned_page_content, seperators, max_chunk_size=400)
        merged_chunks = merge_chunks(chunks, 400, " ")

        for index, chunk in enumerate(merged_chunks):
            all_documents.append(chunk)
            id_counter += 1
            all_meta_datas.append({"page" : i+1})
            all_ids.append(f"Chunk_{id_counter}")

    batch_size = 20
    sleep_between_batches = 65 

    for start in range(0, len(all_documents), batch_size):
        end = start + batch_size
        collection.upsert(
            documents=all_documents[start:end],
            ids=all_ids[start:end],
            metadatas=all_meta_datas[start:end]
        )
        print(f"Upserted batch {start} to {end}")
        time.sleep(sleep_between_batches)

def retrieve(query, collection, re_ranker, n_results=15):
    query_vec = genai_client.models.embed_content(model="gemini-embedding-001", contents=query, 
                                                      config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")).embeddings[0].values
    
    results = collection.query(query_embeddings=[query_vec], n_results=n_results)
    
    ids = results["ids"][0]
    docs = results["documents"][0]
    scores = re_ranker.predict([[query, doc] for doc in docs])

    re_ranked = list(zip(ids, docs, scores))
    return sorted(re_ranked, key=lambda x: x[2], reverse=True)


def get_context(re_ranked, collection, window=2, ratio=0.5, max_anchors=2, keep_top=3):
   
    top_score = re_ranked[0][2]
    if top_score < 0:
        return None

    anchors = [
        int(cid.split("_")[1])
        for cid, text, score in re_ranked[:max_anchors]
        if score >= ratio * top_score
    ]
    numbers = expand_with_neighbors(anchors, window, collection.count())

    top_numbers = [int(cid.split("_")[1]) for cid, text, score in re_ranked[:keep_top]]
    numbers = sorted(set(numbers) | set(top_numbers))
    ids = [f"Chunk_{n}" for n in numbers]

    fetched = collection.get(ids=ids, include=["documents", "metadatas"])
    
    id_to_text = dict(zip(fetched["ids"], fetched["documents"]))
    id_to_page = {cid: meta["page"] for cid, meta in zip(fetched["ids"], fetched["metadatas"])}
    
    texts = [id_to_text[i] for i in ids]
    pages = sorted({id_to_page[i] for i in ids})
    return texts, pages

def answer_query(query, collection, re_ranker):
    re_ranked = retrieve(query, collection, re_ranker)

    result = get_context(re_ranked, collection)

    if result is None:
        return {"status": "not_in_document", "answer": None, "pages": []}

    texts, pages = result

    try:
        answer = generate_answer_LLM(texts, query)
    except errors.ServerError:
        return {"status": "model_busy", "answer": None, "pages": pages}

    return {"status": "ok", "answer": answer, "pages": pages}




if __name__ == "__main__":

    collection, re_ranker = setup()

    ingest(FILE_PATH, collection)

    # query_list= [
    #     "What are the three different ways multi-head attention is used in the Transformer?",
    #     "What optimizer was used to train the model?",
    #     "What regularization techniques were used during training?",
    # ]
    # for query in query_list:
    #         out = answer_query(query, collection, re_ranker)
    #         print(f"\nQ: {query}")
    #         if out["status"] == "ok":
    #             print(out["answer"])
    #             print(f"Sources: pages {out['pages']}")
    #         elif out["status"] == "not_in_document":
    #             print("This question can't be answered from the document.")
    #         else:
    #             print("The model is busy right now, please try again later.")

    
