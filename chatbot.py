from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
os.environ['OPENAI_API_KEY'] = "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"
os.environ['SERPAPI_API_KEY'] = '360d22e4bc0b06f384cdc79db107bd5ef547daa1c1843698dfcff447654b98e5'
st.set_page_config(page_title="joe's chat robot", page_icon=":robot:")
st.header("chat bot for coding")

@st.cache_resource(ttl=10800) 
def create_conversation_chain():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "The following is conversation between a coder and an AI expert in codeing. The AI "
            "provides lots of specific details from its context. If the AI does not know the answer to a "
            "question, it truthfully says it does not know."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
    return conversation


conversation = create_conversation_chain()

col1, col2 = st.columns(2)


input_text = st.text_area(label="输入", placeholder="INPUT...", key="human_input")


output_text = conversation.predict(input=input_text)
if output_text:
    st.write(output_text)
else:
    st.write("fail to get chatbot response")