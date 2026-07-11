import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
page_title="Book RAG Assistant",
page_icon="📚",
layout="wide"
)

st.title("📚 Book RAG Assistant")
st.write("Upload a PDF book and ask questions about it.")

uploaded_file = st.file_uploader(
"Upload PDF",
type=["pdf"]
)

embedding_model = HuggingFaceEmbeddings(
model_name="sentence-transformers/all-MiniLM-L6-v2"
)

prompt = ChatPromptTemplate.from_messages(
[
(
"system",
"""
You are a helpful AI assistant.

```
        Use ONLY the provided context.

        If the answer is not present in the context,
        say:
        'I could not find the answer in the document.'
        """
    ),
    (
        "human",
        """
        Context:
        {context}

        Question:
        {question}
        """
    )
]

)

llm = ChatMistralAI(
model="mistral-small-2506"
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".pdf"
) as tmp_file:

        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("Processing PDF..."):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="temp_chroma_db"
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

st.success("PDF Processed Successfully")

user_question = st.chat_input("Ask a question about the book...")

if user_question:
    with st.chat_message("user"):
        st.write(user_question)

docs = retriever.invoke(user_question)

context = "\n\n".join(
    [doc.page_content for doc in docs]
)

final_prompt = prompt.invoke(
    {
        "context": context,
        "question": user_question
    }
)

response = llm.invoke(final_prompt)

with st.chat_message("assistant"):
    st.write(response.content)