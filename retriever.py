from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

video_id = 'JCOb1w_LTOg'

def create_advanced_retriever(chunks: List[Document]):
    """
    Takes text chunks, indexes them inside a local FAISS database,
    saves the vector store locally on disk, and configures an MMR retriever.
    """
    print("[RETRIEVER] Vectorizing and indexing chunks into local FAISS store..")
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the database index files directly to disk for zero-cost cloud deployment
    print("[RETRIEVER] Saving FAISS database index files locally to disk...")
    vector_store.save_local("faiss_index")

    print("[RETRIEVER] configuring retrieval layer to use MMR..")
    advanced_retriever = vector_store.as_retriever(
        search_type = 'mmr',
        search_kwargs = {
            'k': 6,
            'fetch_k': 20, 
            'lambda_mult': 0.6 
        }
    )
    return advanced_retriever

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    from ingestion import fetch_and_build_semantic_chunks

    url = f'https://www.youtube.com/watch?v={video_id}'
    try:
        chunks = fetch_and_build_semantic_chunks(url)
        retriever = create_advanced_retriever(chunks)

        test_query = 'what is the core lesson?'
        results = retriever.invoke(test_query)
        print(f"\n [SUCCESS] Retriever pulled {len(results)} distinct context chunks")
    except Exception as e:
        print(f"Error occured: {e}")