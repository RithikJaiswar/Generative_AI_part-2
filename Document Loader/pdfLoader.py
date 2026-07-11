from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter #TokenTextSplitter

data = PyPDFLoader(file_path='Document Loader\Generative AI- Basic to Advanced Notes.pdf')

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

docs = data.load()

# print(len(docs)) # Docs has 11 document means 11 page == 11 documents , can see individual page as docs[10]

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)