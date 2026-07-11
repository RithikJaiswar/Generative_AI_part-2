from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()

docs = [
    Document(page_content='gradient descent is optimized algorithms used to minimize loss function'),
    Document(page_content='neural networks are trained using optimization algorithm like gradient descent'),
    Document(page_content='weights updates in machine learning models are done using gradient descent'),
    Document(page_content='support vector machines are supervised learning algorithm'),
]

embeddings = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(docs,embeddings)

retriever = vectorstore.as_retriever()

llm = ChatMistralAI(model='mistral-small-latest')

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

query = 'what is gradient descent ?'

docs = multi_query_retriever.invoke(query)

print('--Retrieved Documents--')

for doc in docs:
    print(doc.page_content)