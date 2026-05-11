from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import LLMChain
import google.generativeai as genai
import streamlit as st
import os,warnings

warnings.filterwarnings("ignore")
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(
    page_title="CodeMentor AI",
    page_icon="💻",
    layout="wide"
)

st.markdown("""
<style>
.stApp{background:#0b0b0b;color:white;}
section[data-testid="stSidebar"]{background:#111111;border-right:1px solid #2a2a2a;}
.hero{text-align:center;margin-bottom:25px;}
.hero h1{font-size:65px;color:white;}
.hero p{color:#bdbdbd;font-size:18px;}
.feature{background:#181818;padding:18px;border-radius:18px;border:1px solid #2a2a2a;margin-bottom:40px;}
.feature h3{color:white;font-size:24px;}
.feature p{color:#d1d5db;}
[data-testid="stChatMessage"]{background:#181818;border-radius:18px;padding:12px;border:1px solid rgba(255,255,255,0.08);}
.stChatInput input{background:#1e1e1e!important;color:white!important;border:1px solid #3a3a3a!important;border-radius:18px!important;}
.footer{text-align:center;color:#9ca3af;margin-top:30px;}
</style>
""",unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <style>
    section[data-testid="stSidebar"]{background:#0f172a;border-right:2px solid #38bdf8;}
    .box{background:#1e293b;padding:15px;border-radius:15px;margin-bottom:15px;color:white;}
    .box:hover{background:#334155;cursor:pointer;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="box">
        <h2>🤖 AI Tutor</h2>
        <p>Learn Coding Easily</p>
    </div>
    <div class="box">
        <h3>💻 Python</h3>
    </div>
    <div class="box">
        <h3>🚀 Projects</h3>
    </div>
    <div class="box">
        <h3>🧠 Interview Prep</h3>
    </div>
    """, unsafe_allow_html=True)
    
    creativity=st.slider("AI Creativity",0.0,1.0,0.5)

st.markdown("""
<div class="hero">
<h1>CodeMentor AI</h1>
<p>Intelligent AI Coding Mentor powered by Gemini & LangChain</p>
</div>
""",unsafe_allow_html=True)

c1,c2,c3=st.columns(3)

with c1:
    st.markdown("<div class='feature'><h3>🐞 Debugging</h3><p>Detect coding errors and get step-by-step fixes instantly.</p></div>",unsafe_allow_html=True)

with c2:
    st.markdown("<div class='feature'><h3>📚 DSA Mentor</h3><p>Learn algorithms, recursion, trees, graphs and dynamic programming.</p></div>",unsafe_allow_html=True)

with c3:
    st.markdown("<div class='feature'><h3>🎯 Interview Coach</h3><p>Practice technical interviews with AI-generated coding questions.</p></div>",unsafe_allow_html=True)

memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)

with open("template.txt","r",encoding="utf-8") as file:
    template=file.read()

prompt=ChatPromptTemplate.from_messages([
    ("system",template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{input}")
])

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=creativity,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

chain=LLMChain(llm=llm,prompt=prompt,memory=memory)

if "messages" not in st.session_state:
    st.session_state.messages=[]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input=st.chat_input(
    "Ask any coding question..."
)

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result=chain.invoke({
                "input":user_input
            })
            response=result['text']
            st.markdown(response)
    st.session_state.messages.append({
        "role":"user",
        "content":user_input
    })
    st.session_state.messages.append({
        "role":"assistant",
        "content":response
    })

st.markdown("""
<div class="footer">
<h4>Powered by Gemini + LangChain + Streamlit</h4>
</div>
""",unsafe_allow_html=True)