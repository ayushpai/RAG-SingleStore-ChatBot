import os
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.singlestoredb import SingleStoreDB

openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["SINGLESTOREDB_URL"] = "<Insert SingleStore Database URL Here>"

# Load text samples
loader = TextLoader("michael_jackson.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

docsearch = SingleStoreDB.from_documents(
    docs,
    embeddings,
    table_name="notebook",  # use table with a custom name
)

query = "What does Michael Jackson have to do with Ronald Reagan?"
docs = docsearch.similarity_search(query)  # Find documents that correspond to the query



print(docs[0].page_content)
