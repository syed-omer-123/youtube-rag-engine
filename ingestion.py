import os
from typing import List
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

def fetch_and_build_semantic_chunks(video_url: str) -> List[Document]:
    """
    Downloads a YouTube transcript and cuts it into pieces based on topic shifts 
    instead of character counts.
    """
    video_id = video_url.split("v=")[-1].split("&")[0]
    print(f"[INGESTION] Processing Video ID: {video_id}")

    try:
        api = YouTubeTranscriptApi()
        raw_transcript = api.fetch(video_id, languages=['en'])
    except Exception as e:
        raise RuntimeError(f"Could not download Transcript. Error: {e}")
    
    full_text = " ".join(
    [entry.text.replace('\n', ' ') for entry in raw_transcript])
    base_document = Document(
        page_content=full_text,
        metadata={'source': video_id, "url": f"https://youtube.com/watch?v={video_id}"}
    )

    print("[INGESTION] Splitting text based on semantic meaning shifts...")
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    semantic_splitter = SemanticChunker(embeddings)

    semantic_chunks = semantic_splitter.split_documents([base_document])
    print(f"[INGESTION] Created {len(semantic_chunks)} smart context chunks.")

    return semantic_chunks

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    url = 'https://www.youtube.com/watch?v=JCOb1w_LTOg'
    try:
        chunks = fetch_and_build_semantic_chunks(url)
        print(f"\n--- Preview of Chunk 1 ---\n{chunks[0].page_content[:250]}...")
    except Exception as e:
        print(f"Error occurred: {e}")