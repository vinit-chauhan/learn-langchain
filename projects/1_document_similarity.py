from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


load_dotenv()


docs = [
    "The quick brown fox jumps over the lazy dog.",
    "A deep learning model can extract complex patterns from data.",
    "Elasticsearch is a powerful tool for distributed search and analytics.",
    "Climate change is impacting ecosystems around the world.",
    "I love hiking in the Rocky Mountains during summer.",
    "Python is a versatile programming language used in many domains.",

    "Apple released a new version of the iPhone last week.",
    "Machine learning algorithms require good quality data to perform well.",
    "The stock market fluctuated wildly after the economic news.",
    "GraphQL enables clients to request exactly the data they need.",
    "My dog loves to chase squirrels in the backyard.",
    "Golang has excellent support for concurrency through goroutines.",

    "Traveling to Japan was one of the best experiences of my life.",
    "The research paper was published in a leading AI journal.",
    "I had pizza for dinner and watched a sci-fi movie.",
    "Docker containers are lightweight and great for microservices.",
    "SpaceX launched another batch of Starlink satellites.",
    "The new restaurant downtown serves amazing vegan food.",

    "React and TypeScript make a solid combination for frontend apps.",
    "The project deadline was moved due to unforeseen circumstances."
]


embedding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=300)

query = 'tell me about an animal'

# Store this to Vector DB
document_embeddings = embedding.embed_documents(docs)


query_embedding = embedding.embed_query(query)

similarities = cosine_similarity(
    np.array([query_embedding]), np.array(document_embeddings))[0]

doc_scores = sorted(list(enumerate(similarities)), key=lambda x: x[1])[::-1]

top_k = 3
for doc_score in doc_scores[:top_k]:
    index, score = doc_score
    print(f'{docs[index]}: {score}')
