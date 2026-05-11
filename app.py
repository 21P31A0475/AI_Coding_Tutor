from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import LLMChain 
import google.generativeai as genai
import os
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(
    page_title="CodeMentor AI",
    page_icon="💻",
    layout="wide"
)

st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right,#0f172a,#1e293b);
    color:white;
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background:#020617;
    border-right:1px solid #334155;
}

/* Main Title */

.main-title{
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#38bdf8;
    margin-bottom:10px;
}

.sub-title{
    text-align:center;
    color:#cbd5e1;
    font-size:20px;
    margin-bottom:30px;
}

/* Feature Cards */

.feature-card{
    background:#111827;
    padding:20px;
    border-radius:16px;
    border:1px solid #334155;
    transition:0.3s;
    margin-bottom:15px;
}

.feature-card:hover{
    transform:scale(1.02);
    border:1px solid #38bdf8;
}

.feature-title{
    color:#38bdf8;
    font-size:22px;
    font-weight:bold;
}

.feature-text{
    color:#d1d5db;
    font-size:15px;
}

/* Footer */

.footer{
    text-align:center;
    color:#94a3b8;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

load_dotenv()

with st.sidebar:

    st.title("💻 CodeMentor AI")

    st.markdown("---")

    st.markdown("## 🚀 Features")

    st.markdown("""
    ✅ Code Debugging  
    ✅ DSA Problem Solving  
    ✅ Coding Interview Preparation  
    ✅ Time Complexity Analysis  
    ✅ Code Explanation  
    ✅ Multi-Language Support  
    ✅ AI Pair Programming  
    ✅ Memory-Based Conversation  
    """)

    st.markdown("---")

    st.markdown("## 🛠 Tech Stack")

    st.markdown("""
    - Gemini 2.5 Flash  Lite 
    - LangChain  
    - Streamlit  
    - Conversation Memory  
    - Prompt Engineering  
    """)

    st.markdown("---")

    st.info(
        "This AI Coding Mentor helps students and developers "
        "improve programming skills interactively."
    )


st.markdown(
    "<div class='main-title'>CodeMentor AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Intelligent Programming Assistant using LLMs</div>",
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class='feature-card'>
        <div class='feature-title'>🐞 Debugging Assistant</div>
        <div class='feature-text'>
            Detect coding errors and get step-by-step fixes instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class='feature-card'>
        <div class='feature-title'>📚 DSA Mentor</div>
        <div class='feature-text'>
            Learn arrays, trees, graphs, recursion, dynamic programming and more.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class='feature-card'>
        <div class='feature-title'>🎯 Interview Coach</div>
        <div class='feature-text'>
            Practice technical interviews with AI-generated coding questions.
        </div>
    </div>
    """, unsafe_allow_html=True)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

with open("template.txt", "r", encoding="utf-8") as file:
    template = file.read()

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


user_input = st.chat_input(
    "Ask any coding question..."
)

if user_input:
    try:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                result = chain.invoke({"input": user_input})
                response = result['text']
                st.markdown(response)

        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    except Exception as e:

        st.error("Failed to generate response. Please try again.")

st.markdown("""
<div class='footer'>
    <h4>Made with ❤️ using Streamlit + Gemini + LangChain</h4>
</div>
""", unsafe_allow_html=True)