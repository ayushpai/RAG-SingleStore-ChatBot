import os
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.singlestoredb import SingleStoreDB
from openai import OpenAI

# Set up API keys and database URL
openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["SINGLESTOREDB_URL"] = "<Insert SingleStore Database URL Here>"

# Load and process documents
loader = TextLoader("openai_documentation.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Generate embeddings and create a document search database
embeddings = OpenAIEmbeddings()
docsearch = SingleStoreDB.from_documents(docs, embeddings, table_name="notebook")

# Initialize OpenAI client
client = OpenAI()

# Chat loop
while True:
    # Get user input
    user_query = input("\nYou: ")

    # Check for exit command
    if user_query.lower() in ['quit', 'exit']:
        print("Exiting chatbot.")
        break

    # Perform similarity search
    docs = docsearch.similarity_search(user_query)
    if docs:
        context = docs[0].page_content

        # Generate response using OpenAI GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Context: " + context},
                {"role": "user", "content": user_query}
            ],
            stream=True,
            max_tokens=500,
        )

        # Output the response
        print("AI: ", end="")
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")

    else:
        print("AI: Sorry, I couldn't find relevant information.")
