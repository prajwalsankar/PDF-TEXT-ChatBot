import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
from langchain_community.vectorstores import FAISS
import pickle
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = genai.GenerativeModel('gemini-pro')

with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powred Chatbot built using:
    [streamlit](https://streamlit.io/)
    Langchain
    OpenAI
    GeminiAI
    ''')
    add_vertical_space(5)


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def main():
    st.write("Chat With PDF....")
    pdf = st.file_uploader('Upload your PDF',type='pdf')
    st.write(pdf)

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # st.write(pdf_reader)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 10000,
            chunk_overlap=1000,
            length_function = len
        )
        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)
        # st.write(text)
        #embeddings
        # get embeddings for each chunk
        # embeddings = GoogleGenerativeAIEmbeddings(
        # model="models/embedding-001")

        # vectorstore = FAISS.from_texts(chunks,embedding= embeddings)
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pk1"):
            try : 
                with open(f"{store_name}.pk1","rb") as f:
                    vectorstore = print(pickle.load(f))
                # st.write(vectorstore)
                st.write('Embeddings Loaded from the Disk')
            except EOFError : 
                return dict()
            
        else:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.from_texts(chunks,embedding= embeddings)
            with open(f"{store_name}.pk1","wb") as f:
                pickle.dump(vectorstore,f)
            # st.write(vectorstore)
            st.write('Embedding Validation Completed')

        #Accept User Questions/Queries
        query = st.text_input("Ask Questions about your PDF File:") 
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_texts(chunks,embedding= embeddings)      
        # print(vectorstore)
        if query:
            st.write(query)
            docs = vectorstore.similarity_search(query=query,k=3)
            # st.write(docs)
            
            model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3
                                   )
            # prompt = PromptTemplate(template=prompt_template,
                            # input_variables=["context", "question"])
            # chain = load_qa_chain(llm=model, chain_type="stuff")
            # response = chain.run({"input_documents": docs, "question": query})

            
            chain = get_conversational_chain()

            response = chain({"input_documents": docs, "question": query}, return_only_outputs=True, )

            print(response)
            st.write(response)

if __name__ =='__main__':
    main()
    # st.set_option('server.enableCORS', True) 
    