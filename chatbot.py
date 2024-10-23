import os
os.environ['OPENAI_API_KEY'] = ""

import streamlit as st
# Set the title using StreamLit
st.title(' Infinio Learn ')
input_text = st.text_input('Ask your doubt: ') 
import streamlit as st
# from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
# from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
# from huggingface_hub import hf_hub_download
from InstructorEmbedding import INSTRUCTOR
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# import os
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
# import time
from langchain.chat_models import ChatOpenAI

def get_vectorstore():
    load_dotenv()
    embeddings = HuggingFaceInstructEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore1=FAISS.load_local("vector_store",embeddings,allow_dangerous_deserialization=True)
    return vectorstore1

def rqa(vectorstore1):
    retriever = vectorstore1.as_retriever(search_type="similarity", search_kwargs={"k":3})
    llms = ChatOpenAI(openai_api_key="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",temperature=0)
    rqa=RetrievalQA.from_chain_type(llm=llms,retriever=retriever,return_source_documents=True)
    return rqa

# Display the output if the the user gives an input
global answer1
load_dotenv()
vector=get_vectorstore()
ret=rqa(vector)
if input_text: 
    answer1=ret.invoke(input_text)
    st.write(answer1['result']) 
    with st.expander('Source '): 
        st.info(answer1['source_documents'][0].metadata)