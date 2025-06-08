from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=10)

docs = [
    'What is a dog?',
    'What is a cat?',
    'Cat is an Animal.',
    'Dog is an Animal.',
    'Lion is a type of cat',
    'Lion is not a dog',
    'Pomerian is a type of dog'
]

vectors = embeddings.embed_documents(docs)

for idx, vector in enumerate(vectors):
    print(f'Vector for #{docs[idx]}: {vector}')
