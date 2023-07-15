from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts import PromptTemplate
import os
from langchain.chains import LLMChain
os.environ['OPENAI_API_KEY'] = "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"
os.environ['SERPAPI_API_KEY'] = '360d22e4bc0b06f384cdc79db107bd5ef547daa1c1843698dfcff447654b98e5'

"""
demo 1 usage of llm, text-in --> text-out
"""
llm = OpenAI(temperature=0.9)
# input_text = "What would be a good company name for a company that makes colorful socks?"
# output_text = llm.predict(input_text)
# print(output_text)

"""
demo2 
Chat models are a variation on language models. While chat models use language models under the hood,
the interface they expose is a bit different: rather than expose a "text in, text out" API, 
they expose an interface where "chat messages" are the inputs and outputs.
You can get chat completions by passing one or more messages to the chat model. The response will be a message. 
The types of messages currently supported in LangChain are AIMessage, HumanMessage, SystemMessage, and ChatMessage --
ChatMessage takes in an arbitrary role parameter. Most of the time, you'll just be dealing with HumanMessage, AIMessage, 
and SystemMessage.
"""
# chat = ChatOpenAI(temperature=0)
# input_message = [SystemMessage(content="You are an expert of C++"), HumanMessage(content="Explain why C++ is better than python")]
# output_message = chat.predict_messages(input_message)
"""
demo 3 usage of templete
"""
# prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
# chain = LLMChain(llm=llm, prompt=prompt)
# prompt_text = prompt.format(product="colorful socks")
# output_text = chain.run("colorful socks")
# print(output_text)

"""
demo4 templete for chain
"""
# chat = ChatOpenAI(temperature=0)

# template = "You are a helpful assistant that translates {input_language} to {output_language}."
# system_message_prompt = SystemMessagePromptTemplate.from_template(template)
# human_template = "{text}"
# human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
# chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# chain = LLMChain(llm=chat, prompt=chat_prompt)
# output_text = chain.run(input_language="English", output_language="French", text="I love programming.")
# print(output_text)

"""
demo 5 usage of an agent
"""
# from langchain.agents import load_tools
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType

# # First, let's load the language model we're going to use to control the agent.
# chat = ChatOpenAI(temperature=0)

# # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
# llm = OpenAI(temperature=0)
# tools = load_tools(["serpapi", "llm-math"], llm=llm)

# # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
# agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# input_text = "deepstream中1920x1080大小的视频流数据缓存格式是什么？假设缓存120帧数据，会需要占用多少空间？"
# # Now let's test it out!
# output_text = agent.run(input_text)
# print(output_text)

# demo 6 usage of memory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "The following is a friendly conversation between a human and an AI. The AI is talkative and "
        "provides lots of specific details from its context. If the AI does not know the answer to a "
        "question, it truthfully says it does not know."
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

output_text = conversation.predict(input="please remember the number 10")
print(output_text)
output_text = conversation.predict(input="tell me which number i told you")
print(output_text)
