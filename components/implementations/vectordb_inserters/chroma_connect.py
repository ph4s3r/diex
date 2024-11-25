#####
#   about: Script to connect to a chroma server
#  author: Peter Karacsonyi <peterkaracsonyi85@gmail.com>
#    date: 22 Nov 2024
# license: GNU General Public License, version 2
#####

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
chroma_client = chromadb.HttpClient(
    host="chroma.d3eyekfegubub9fz.switzerlandnorth.azurecontainer.io",
    port=8000
)
print("trying to ping the server")
print("heartbeat received:", chroma_client.heartbeat())

print("running chroma version:", chroma_client.get_version())
# print("settings:", chroma_client.get_settings())
print("get_max_batch_size:", chroma_client.get_max_batch_size())

collections = chroma_client.list_collections()

for i, coll in enumerate(collections):
    print(i, coll)

# print("peeking into a collection: ", collections[0].peek())

st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="dunzhang/stella_en_1.5B_v5")

queries = [
    "How to set up a GitHub Actions workflow to register an API?",
]
c = collections[0]

c._embedding_function = st_ef

results = c.query(
    query_texts=queries,        # Chroma will embed this for you
    n_results=2                 # how many results to return
)
print(results)



# query:

# def query(
#         query_embeddings: Optional[OneOrMany[Embedding]] = None,
#         query_texts: Optional[OneOrMany[Document]] = None,
#         n_results: int = 10,
#         where: Optional[Where] = None,
#         where_document: Optional[WhereDocument] = None,
#         include: Include = ["metadatas", "documents",
#                             "distances"]) -> QueryResult





