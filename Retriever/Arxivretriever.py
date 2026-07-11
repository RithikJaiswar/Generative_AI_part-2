import arxiv
from langchain_core.documents import Document

client = arxiv.Client()

search = arxiv.Search(
    query="large language model",
    max_results=2,
    sort_by=arxiv.SortCriterion.Relevance
)

docs = []
for result in client.results(search):
    docs.append(Document(
        page_content=result.summary,
        metadata={
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "published": result.published,
            "url": result.entry_id
        }
    ))

for doc in docs:
    print(doc.metadata["title"])
    print(doc.page_content[:300])
    print("---")


# from langchain_community.retrievers import ArxivRetriever

# # create the retrievers
# retriever = ArxivRetriever(
#     load_max_docs=3 , # number of paper to retrieve
#     load_all_available_meta=True
# )

# # query arxiv
# docs = retriever.invoke('large language model')

# # print results
# for i, docs in enumerate(docs):
#     print(f'\nResult{i+1}')
#     print('Title:',docs.metadata.get('Title'))
#     print('Authors:',docs.metadata.get('Authors'))
#     print('Summary:',docs.page_content[:500]) # print 500 character