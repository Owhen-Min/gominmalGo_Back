from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

def init_pinecone():
    with open('./env/pinecone_key', 'r') as file:
        key = file.read().strip()
    pinecone.init(api_key=key, environment="us-east-1")

def search_emotion(query):
    init_pinecone()
    model_dir = "./model"
    latest_model = max([d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))])
    model = SentenceTransformer(os.path.join(model_dir, latest_model))
    index = pinecone.Index(latest_model)

    query_vector = model.encode(query).tolist()
    results = index.query(query_vector, top_k=1, include_metadata=True)
    return results

def search_wellness(query):
    init_pinecone()
    index_name = "wellness_model"
    index = pinecone.Index(index_name)
    model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

    query_vector = model.encode(query).tolist()
    results = index.query(query_vector, top_k=1, include_metadata=True)
    return results
