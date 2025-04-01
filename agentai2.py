import streamlit as st
import requests
import wikipedia
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI 
from langchain.chains import RetrievalQA

import os

# Load API keys securely from Streamlit secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY_1", "")
weather_api_key = st.secrets.get("OPENWEATHER_API_KEY", "")
flight_api_key = st.secrets.get("FLIGHT_API_KEY", "")

if not openai_api_key:
    st.error("Missing OpenAI API key. Please add it to Streamlit Secrets.")
    st.stop()

# 🟢 Cache document processing (FAISS embedding)
@st.cache_resource(show_spinner=False)
def load_and_vectorize_pdfs(pdf_folder):
    """Loads and vectorizes PDFs from the specified folder (cached)."""
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # Explicit API key
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# 🟢 Cache weather data for 30 minutes
@st.cache_data(ttl=1800)  
def get_weather(city):
    """Fetch real-time weather information (cached)."""
    if not weather_api_key:
        return "Weather API key is missing."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return f"{data['weather'][0]['description'].capitalize()}, Temperature: {data['main']['temp']}°C"
    return "Weather data not available."

# 🟢 Cache flight data for 1 hour
@st.cache_data(ttl=3600)  
def get_flight_details(origin, destination, date):
    """Fetch real-time flight details (cached)."""
    if not flight_api_key:
        return "Flight API key is missing."

    url = f"https://api.flightapi.io/search/{flight_api_key}/{origin}/{destination}/{date}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "flights" in data and data["flights"]:
            flight = data["flights"][0]
            return f"Flight {flight['flight_number']} from {flight['departure']} to {flight['arrival']} on {flight['date']} at {flight['time']}"
    return "Flight data not available."

# Wikipedia Lookup
def search_wikipedia(query):
    """Search Wikipedia for general knowledge if FAISS has no answer."""
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found: {e.options[:3]}"
    except wikipedia.exceptions.PageError:
        return None  # No Wikipedia page found

def main():
    st.set_page_config(page_title="Agentic AI Travel Assistant", page_icon="🌍", layout="centered")
    
    st.title("🌍 Agentic AI Travel Assistant")
    st.write("Hello! I'm your AI travel assistant, ready to help you with:")

    # Feature overview
    features = {
        "📍 Places": "Get details about cities, landmarks, and hidden gems.",
        "🌦️ Weather": "Real-time weather forecasts for any location.",
        "🍽️ Cuisines": "Discover local and international food specialties.",
        "🏝️ Destinations": "Explore top tourist attractions and experiences.",
        "🛫 Travel Packages": "Find the best travel deals from our brochures.",
        "✈️ Flights": "Check flight details and availability."
    }
    for icon, description in features.items():
        st.markdown(f"- {icon}: {description}")

    # User query input
    user_query = st.text_input("What would you like to know?", "Best tourist spots in Paris")

    # Load FAISS vectorstore (cached)
    pdf_folder = "brochures"
    vectorstore = load_and_vectorize_pdfs(pdf_folder)
    retriever = vectorstore.as_retriever()

    # Initialize OpenAI LLM
    llm = OpenAI(api_key=openai_api_key)

    # RetrievalQA with explicit chain type for better performance
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    if st.button("Get Information"):
        with st.spinner("Fetching details..."):
            response = None  # Initialize response variable

            if "weather" in user_query.lower():
                city = user_query.split(" in ")[-1]
                response = get_weather(city)
            elif "flight" in user_query.lower():
                parts = user_query.split(" from ")[-1].split(" to ")
                if len(parts) == 2:
                    origin, destination = parts
                    date = "2025-04-01"  # Placeholder date
                    response = get_flight_details(origin.strip(), destination.strip(), date)
                else:
                    response = "Please specify the flight origin and destination."
            else:
                # Step 1: Try FAISS Vector Store First
                response = qa_chain.run(user_query)

                # Step 2: If FAISS returns no answer, try Wikipedia
                if "I don't know" in response or len(response.strip()) < 5:
                    wiki_result = search_wikipedia(user_query)
                    if wiki_result:
                        response = wiki_result
                        response += "\n\nFor exclusive travel packages to these destinations, contact Margie’s Travel! ✈️🌍"
                
                # Step 3: If Wikipedia also fails, use OpenAI LLM
                if response is None or "I don't know" in response:
                    response = llm(user_query)
                    response += "\n\nFor exclusive travel packages to these destinations, contact Margie’s Travel! ✈️🌍"
            
            st.success(f"**{user_query}**: {response}")

if __name__ == "__main__":
    main()
