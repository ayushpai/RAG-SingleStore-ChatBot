import os
import openai
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.singlestoredb import SingleStoreDB

# Create prompt template
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

from retrival import docsearch

openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["SINGLESTOREDB_URL"] = "admin:Test1234@svc-7301c603-3097-4c72-bc10-7881e89ff282-dml.aws-virginia-6.svc.singlestore.com:3306/RAGTester"


prompt_template = """
Use the following pieces of context to answer the question at the end. If
you're not sure, just say so. If there are potential multiple answers,
summarize them as possible answers.
{context}
Question: {question}
Answer:
"""

PROMPT = PromptTemplate(template=prompt_template,
input_variables=["context", "question"])

# Initialize RetrievalQA object
qa_chain = load_qa_chain(OpenAI(), chain_type="stuff")
chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name='gpt-4-0613'),
                                 chain_type="stuff",
                                 retriever=docsearch.as_retriever(),
                                 chain_type_kwargs=chain_type_kwargs)

# Query the data
query = "What did Michael Jackson do with George Lucas?"
qa.run(query)