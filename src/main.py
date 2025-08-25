import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings 
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = 'crm-faq-system'
csv_file_path = 'data/codebasics_faqs.csv'

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=gemini_api_key)

embeddings = HuggingFaceEmbeddings(
        model_name='hkunlp/instructor-large',
        encode_kwargs={
            "query_instruction": "Represent the query for retrieval: "
        }
    )
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)


def create_vectordb():
    loader = CSVLoader(file_path='data/codebasics_faqs.csv', encoding="latin-1", source_column='prompt')
    data = loader.load()

    vectordb = PineconeVectorStore(index=index, embedding=embeddings)
    vectordb.add_documents(documents=data)


def get_qa_chain():
    vectordb = PineconeVectorStore.from_existing_index(
        index_name=pinecone_index_name,
        embedding=embeddings
    )

    retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs = {"score_threshold": 0.7}
    )

    template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    prompt = ChatPromptTemplate.from_template(template=template)

    
    chain = (
        {'query': RunnablePassthrough()} 
        | RunnablePassthrough.assign(
            context=itemgetter('query') | retriever,
            question=itemgetter('query')
        )
        | prompt 
        | llm 
        | StrOutputParser())
    
    
    return chain 

if __name__ == "__main__":
    chain = get_qa_chain()
    print(chain.invoke("I have a MAC computer, can I use power BI on it?"))

    

