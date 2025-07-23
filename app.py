import os 
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env
load_dotenv()

# Optional LangChain tracking (only if using LangSmith or analytics)
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY', '')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_OLLAMA_PROJECT'] = os.getenv('LANGCHAIN_OLLAMA_PROJECT', '')

# Setup Streamlit UI
st.set_page_config(page_title="Langchain + gemma3:1b Demo")
st.title("Langchain Demo With gemma3:1b Model")

# User input
input_text = st.text_input("Ask a question")

# Setup LangChain prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's question."),
    ("user", "Question: {question}")
])

# Setup LangChain components
llm = Ollama(model="gemma3:1b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Run the chain
if input_text:
    with st.spinner("Thinking..."):
        try:
            response = chain.invoke({"question": input_text})
            st.success("Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
