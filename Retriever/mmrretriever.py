from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
 
docs = [
    Document(page_content='Gradient Descent is an optimization algorithm used in machine learning'),
    Document(page_content='Gradient descent minimizes the loss function'),
    Document(page_content='gradient descent is an optimization that minimizes the loss function'),
    Document(page_content='neural networks use gradient descent for training '),
    Document(page_content='support vector machines are supervised learning algorithms')
]

embedding = HuggingFaceEmbeddings()

vector_store = Chroma.from_documents(docs,embedding)

# using Similarity retriever
similarity_retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k':3}
)

print('Similarity search')

similarity_docs = similarity_retriever.invoke('what is gradient descent')

for docs in similarity_docs:
    print(docs.page_content)

# using MMR retriever -> skips repeated value , give a proper format
mmr_retriever = vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={'k':3}
)

print('MMR results')

mmr_docs = mmr_retriever.invoke('what is gradient descent')

for docs in mmr_docs:
    print(docs.page_content)

