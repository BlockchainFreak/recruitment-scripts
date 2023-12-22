import pinecone
import dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from database import create_db_session
from models import StructuredResume

dotenv.load_dotenv()

embeddings = OpenAIEmbeddings()

DATABASE_URL = os.environ["DATABASE_URL"]

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="gcp-starter")

# JD goes here
query="React Developer, JS, HTML, Web Experience, 5 years"

query_embedding = embeddings.embed_query(query)

index = pinecone.Index(index_name="resumes")

results = index.query(
    vector=query_embedding,
    top_k=3
).matches

session = create_db_session()

top_resumes = []
for result in results:
    resume = session.query(StructuredResume).filter_by(id=result.id).first().structured
    top_resumes.append(resume)

print(top_resumes)