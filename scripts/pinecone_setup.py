# Replace with your Pinecone API key and environment
API_KEY = "PINECONE_API_KEY_REMOVED"
index_name = "test"
cloud =  'aws'
region = 'us-east-1'
model_name = 'dunzhang/stella_en_1.5B_v5'





import time
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings

pc = Pinecone(api_key=API_KEY)
spec = ServerlessSpec(cloud=cloud, region=region)

# with langchain
# https://python.langchain.com/docs/integrations/vectorstores/pinecone/

############## DEFINE MODEL ###############

model_kwargs = {
    'device': 'cuda',
    'trust_remote_code':True
    }

query_prompt_name = 's2p_query'

encode_kwargs = {
                    'normalize_embeddings': True,
                    'prompt_name': query_prompt_name
}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
############## CREATE INDEX ###############

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=spec,
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# See that it is empty
print("Index before upsert:")
print(pc.Index(index_name).describe_index_stats())
print("\n")
