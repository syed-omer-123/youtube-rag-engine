import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from ingestion import fetch_and_build_semantic_chunks
from retriever import create_advanced_retriever

class ExecutiveSummary(BaseModel):
    video_title: str = Field(description="a concise, professional title summarizing the video's core focus")
    main_takeaway: str = Field(description='a primary macro lesson or thesis delivered by speaker')
    key_pillars: list[str] = Field(description="A detailed list of distinct core strategies or pillars discussed, extracted from context")
    actionable_steps: list[str] = Field(description="Practical, Operational advice or steps derived staright from the speaker's points.")

def run_production_pipeline(video_url: str):
    print("\n== STARTING PRODUCTION PIPELINE ==")
    chunks = fetch_and_build_semantic_chunks(video_url)
    retriever = create_advanced_retriever(chunks)

    core_query = 'what is the main takeaways, strategies,lessons and actionable insights discussed in the video?'
    retrieved_docs = retriever.invoke(core_query)
    context_text = "\n\n".join([f"--Context Block--\n{doc.page_content}" for doc in retrieved_docs])

    prompt = ChatPromptTemplate.from_messages([
        ('system', ("You are an expert executive analyst. Your task is to process the following retrieved transcript context "
            "from a YouTube video and extract a deep, highly structured executive summary.\n\n"
            "CRITICAL DIRECTIVE:\n"
            "- Depend ONLY on the provided context below. Do not use outside knowledge.\n"
            "- Maintain an objective, straightforward tone. Do not use filler or fluff.\n\n"
            "Retrieved Transcript Context:\n{context}")),
        ('human', "Generate the structured executive summary based exactly on our schema definition.")
    ])

    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
    structured_llm = llm.with_structured_output(ExecutiveSummary)
    chain = prompt | structured_llm
    output_json = chain.invoke({'context': context_text})
    return output_json

def answer_dynamic_query(user_question: str) -> str:
    """
    Loads the pre-saved local FAISS database index from disk,
    performs an MMR context search, and queries GPT-4o-mini for a natural language answer.
    """
    print(f"[DYNAMIC QUERY] Loading local index to answer: '{user_question}'")
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    
    # Load index directly from disk (Zero Ingestion / Zero Embedding Cost)
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    retriever = vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 4, 'fetch_k': 20, 'lambda_mult': 0.6}
    )
    
    retrieved_docs = retriever.invoke(user_question)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ('system', ("You are an expert conversational AI assistant. Answer the user's question accurately "
            "using ONLY the provided video transcript context below. If the answer cannot be found, "
            "politely state that the video context does not contain that information.\n\n"
            "Context:\n{context}")),
        ('human', "{question}")
    ])
    
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
    chain = prompt | llm
    
    response = chain.invoke({"context": context_text, "question": user_question})
    return response.content

if __name__ == '__main__':
    load_dotenv()
    target_url = "https://www.youtube.com/watch?v=JCOb1w_LTOg"
    try:
        final_summary = run_production_pipeline(target_url)
        print(final_summary.model_dump_json(indent=4))
    except Exception as e:
        print(f"\n[PIPELINE CRASHED] Error: {e}")