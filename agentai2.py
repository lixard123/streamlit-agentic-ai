# -*- coding: utf-8 -*-
"""agentai2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12GDsbsuHD3C3yeeIsgy1gMGrxxiKGkLg
"""

#!pip install streamlit
#!pip install langchain
#!pip install openai
#!pip install python-dotenv
#!pip install langchain_community
#!pip install -U langchain langchain-openai
#!pip install requests
#!pip install wikipedia
#!pip install python-dotenv

import streamlit as st
import requests
import wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Set Streamlit Page Config
st.set_page_config(page_title="Agentic AI", layout="centered")

# API Key Handling
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = st.secrets.get("OPENAI_API_KEY", "")

if not st.session_state["openai_api_key"]:
    st.warning("Please add your OpenAI API key in Streamlit Secrets to proceed.")

# Initialize OpenAI LLM
llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=st.session_state["openai_api_key"])

# Define Functions for Tools
def get_weather_updated(query):
    """Fetch weather from wttr.in"""
    city = query.replace(" ", "+")
    url = f"https://wttr.in/{city}?format=%C+%t"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.text.strip()
    except requests.RequestException:
        return "Sorry, I couldn't retrieve the weather information. Try again later."

def search_wikipedia_updated(query):
    """Fetch summary from Wikipedia"""
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found: {', '.join(e.options[:5])}. Please specify."
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'."
    except Exception as e:
        return f"Error: {str(e)}"

# Define Agent Tools
weather_tool = Tool(name="Weather Tool", func=get_weather_updated, description="Get live weather updates")
wiki_tool = Tool(name="Wikipedia Tool", func=search_wikipedia_updated, description="Fetch information from Wikipedia")

# Initialize Agent
tools = [weather_tool, wiki_tool]
agent = initialize_agent(
    tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Streamlit UI
st.title("🌟 Agentic AI - Chat with an Intelligent Agent")

# User Input
query = st.text_input("Enter your query:", "")

if st.button("Run Agent"):
    if not query:
        st.warning("Please enter a query to proceed.")
    else:
        with st.spinner("Processing..."):
            try:
                response = agent.run(query)
                st.success("✅ Response Generated")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("💡 Built with [LangChain](https://python.langchain.com/) & [Streamlit](https://streamlit.io/)")
