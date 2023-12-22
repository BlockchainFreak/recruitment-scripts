import os
import json
import time
import openai
import threading
import pinecone
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from schema import CandidateRecord
from utils import extract_text_from_pdf, count_gpt_tokens, hash_text, clean_text
from database import create_db_session
from models import StructuredResume
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

"""
Before running this script, make sure you have the following environment variables set:
- OPENAI_API_KEY

You can set them by running the following commands in your terminal:
"""

resumes_directory = "C:/Users/a/Projects/Gaper/hybrid-matcher/TestingCVsForSRIR"

# model = OpenAI()

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def text_to_json(text: str) -> dict:
    """
    Converts the given text into JSON format using OpenAI's GPT-3.5 Turbo model.
    Retries with exponential backoff if the API limit is exceeded.

    Args:
        text (str): The input text to be converted into JSON.

    Returns:
        dict: The JSON representation of the input text.
    """
    completion = completion_with_backoff(
        model="gpt-3.5-turbo-0613",
        functions=[CandidateRecord.openai_schema],
        function_call={"name": CandidateRecord.openai_schema["name"]},
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": f"Extract the information from the resume in json schema \n###\n{text}\n###\n"
            },
        ],
    )

    json_text = completion.choices[0].message.function_call.arguments
    return json.loads(json_text)

log_lock = threading.Lock()
icr_lock = threading.Lock()
num_processed = 0

def process_resume(resume):
    try:
        # Create a new session for each thread to avoid sharing the session across threads
        local_session = create_db_session()  # You need to define this function based on your DB setup

        content = extract_text_from_pdf(resume)
        content = clean_text(content)
        tokens = count_gpt_tokens(content)
        hash = hash_text(content)

        if local_session.query(StructuredResume).filter_by(id=hash).first() is not None:
            print(f"Resume with hash {hash} already exists in the database")
            local_session.close()
            safe_log("resumes.logs", f"Resume: {resume}, Token count: {tokens}, Hash: {hash}, Reason: Already exists in the database")
            return

        candidate = text_to_json(content)
        
        local_session.add(StructuredResume(id=hash, content=content, structured=candidate))
        local_session.commit()
        local_session.close()

        with icr_lock:
            global num_processed
            num_processed += 1
            print(f"Processed {num_processed} resumes")
        
        safe_log("resumes.logs", f"Resume: {resume}, Token count: {tokens}, Hash: {hash}")
    except Exception as e:
        safe_log("resumes.logs", f"Resume: {resume}, Token count: {tokens}, Hash: {hash}, Reason: {e}")
    
def safe_log(log_file: str, messages: list[str]):
    with log_lock:
        with open(log_file, "a") as log:
            log.write(("*" * 80) + "\n")
            log.write(messages + "\n")
            log.write(("*" * 80) + "\n")

def add_structured_json():
    resumes = os.listdir(resumes_directory)
    resumes = [os.path.join(resumes_directory, resume) for resume in resumes]

    with ThreadPoolExecutor(max_workers=12) as executor:
        future_to_resume = {executor.submit(process_resume, resume): resume for resume in resumes}
    
        for future in as_completed(future_to_resume):
            future.result()

def add_embeddings():
    session = create_db_session()
    resumes = session.query(StructuredResume).all()
    session.close()

    contents = [resume.content for resume in resumes]
    ids = [resume.id for resume in resumes]
    metadatas = [resume.structured for resume in resumes]

    pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="gcp-starter")
    Pinecone.from_texts(
        index_name="resumes",
        embedding=OpenAIEmbeddings(),
        ids=ids,
        texts=contents,
        # metadatas=metadatas,
    )
    print("Added embeddings to the database")

if __name__ == "__main__":
    add_structured_json()
    add_embeddings()