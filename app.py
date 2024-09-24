import os
import requests
import streamlit as st 
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS

def validate_hf_key(api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    # Make a simple test request
    response = requests.post(
        "https://api-inference.huggingface.co/models/gpt2",
        headers=headers,
        json={"inputs": "test"}
    )
    return response.status_code == 200


st.title("RAG on PDFs using LangChain,Huggingface and Groq")

if "hf_api_key" not in st.session_state:
    st.session_state.hf_api_key = ""

with st.sidebar:
    session_id= st.text_input("Session ID", value='default_session')
    st.session_state.hf_api_key = st.text_input("Huggingface API Key", placeholder="", type="password", key="hface_api_key")
    groq_api_key = st.text_input("GroQ API Key", placeholder="", type="password", key="groq_api_key")

if st.session_state.hf_api_key:
    if validate_hf_key(st.session_state.hf_api_key):
        if groq_api_key:
            try:
                with st.spinner("Processing"):
                    os.environ["HF_TOKEN"] = st.session_state.hf_api_key  
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    
                    llm = ChatGroq(groq_api_key=st.session_state.groq_api_key,  model="Gemma2-9b-It")
                                
                    if "session" not in st.session_state:
                        st.session_state.store = {}
                        
                    uploaded_file = st.file_uploader("Upload your PDF file(s) here to extract text. You can upload multiple files at once.", type="pdf", accept_multiple_files=True)
                    if uploaded_file:
                        documents = []
                        for uploaded_file in uploaded_file:
                            temp_pdf = "./temp.pdf"                            
                            with open(temp_pdf, "wb") as f:
                                f.write(uploaded_file.getvalue())
                                file_name = uploaded_file.name
                                        
                            loader = PyPDFLoader(temp_pdf)
                            doc = loader.load()
                            documents.extend(doc)
                                        
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
                        chunks = text_splitter.split_documents(documents)
                        vector_store = FAISS.from_documents(chunks, embeddings)
                        retriever = vector_store.as_retriever()
                                    
                        contextualize_system_prompt = """
                        Given a chat history and the latest user question which might 
                        reference context in the chat histry, formulate a standalone question
                        which can be understood without the chat histor.Do NOT answer the question,
                        just reformulate the question if needed otherwise return it as it is.
                        """
                        contextualize_question_prompt = ChatPromptTemplate.from_messages(
                            [
                                ("system", contextualize_system_prompt),
                                MessagesPlaceholder("chat_history"),
                                ("human", "{input}")
                            ]
                        )
                                    
                        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_question_prompt)
                                    
                        system_prompt = """You are a question-answering assistant.
                        Answer the questions only using the context provided. If 
                        you don't know the answer say that you don't know the answer and
                        ask if they have any other question. Answer concise using 4 sentences max.
                        \n\n
                        {context}"""
                                    
                        prompt = ChatPromptTemplate(
                            [
                                ("system", system_prompt),
                                MessagesPlaceholder("chat_history"),
                                ("human", "{input}"),
                            ]
                        )
                                    
                        qa_chain = create_stuff_documents_chain(llm, prompt)
                        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
                                    
                        def get_session_history(session_id:str)-> BaseChatMessageHistory:
                            if session_id not in st.session_state.store:
                                st.session_state.store[session_id] = ChatMessageHistory()
                            return st.session_state.store[session_id]
                                    
                        conversational_rag_chain = RunnableWithMessageHistory(
                            rag_chain,
                            get_session_history,
                            input_messages_key='input',
                            history_messages_key='chat_history',
                            output_messages_key='answer'
                        )
                                    
                                    
                        user_input = st.text_input("Ask a question")
                        if user_input:
                            if not groq_api_key.strip():
                                st.error("GroQ API key is missing. Please enter it again.")
                            else:
                                session_history = get_session_history(session_id)
                                output = conversational_rag_chain.invoke(
                                    {'input': user_input},
                                    config = {"configurable": {"session_id": session_id}}
                                )
                                                
                                st.write("Assistant's response:", output['answer'])
            except Exception as e:
                st.error("Invalid GroQ API Key. Please enter it again.")
        else:
            st.info("Please enter your GroQ API Key")        
    else:
        st.error("Invalid Huggingface API key. Please enter it again.")
        st.session_state.hf_api_key = ""  
else:
    st.info("Please enter your Huggingface API Key to begin using the RAG system.")

    