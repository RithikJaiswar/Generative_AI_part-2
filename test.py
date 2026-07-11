from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    # separator='' / if you want to add manually any separator
    chunk_size=10,
    chunk_overlap=1
)

data = TextLoader('Document Loader\deeplearning.txt')

docs = data.load()

chunks = splitter.split_documents(docs)

print(len(chunks)) # if dont want to see meta data use .page_content