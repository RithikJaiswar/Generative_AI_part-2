from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embedding_model = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

vectorstore = Chroma.from_documents(
    documents = docs,
    embedding= embedding_model,
    persist_directory= "chroma-db"
)

# vectorstore is not responsible for your answers , responsibe for retrieve for your information , for answer llm is responsible 
result = vectorstore.similarity_search('what is used for data analysis?',k=2) # k how output you want

for r in result:
    print(r)

retriever = vectorstore.as_retriever()

docs = retriever.invoke('Explain the deep learning')

for d in docs:
    print(d.page_content)