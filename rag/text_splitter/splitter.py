from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader


loader = TextLoader('rag/text_splitter/data.txt')
text = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)
sep_text = splitter.split_documents(text)

print('CharacterTextSplitter:\n', sep_text[0].page_content)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)
sep_text = splitter.split_documents(text)

print('RecursiveCharacterTextSplitter:\n', sep_text[0].page_content)


splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=20
)
loader = TextLoader('projects/3_resume_score.py')
text = loader.load()
sep_text = splitter.split_documents(text)

print('RecursiveCharacterTextSplitter:\n', sep_text[0].page_content)
