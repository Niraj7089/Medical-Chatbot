from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings  # make sure this works
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

# -----------------------
# Initialize Flask app
# -----------------------
app = Flask(__name__)

# -----------------------
# Load environment variables
# -----------------------
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')  # if needed
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -----------------------
# Load embeddings
# -----------------------
print("Loading embeddings...")
embeddings = download_hugging_face_embeddings()
print("Embeddings loaded.")

# -----------------------
# Connect to Pinecone index
# -----------------------
index_name = "medical-chatbot"
print(f"Connecting to Pinecone index: {index_name}")
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# -----------------------
# Create retriever
# -----------------------
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# -----------------------
# Ollama local LLM
# -----------------------
chatModel = Ollama(
    model="phi3:mini",  # lightweight local LLaMA model
    temperature=0
)

# -----------------------
# Create prompt template
# -----------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# -----------------------
# Create RAG chain
# -----------------------
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# -----------------------
# Flask routes
# -----------------------
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"].strip().lower()
    print("User Input:", msg)

    # --- Handle greetings ---
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if msg in greetings:
        return "Hello! How can I assist you today?"

    # --- Handle polite phrases ---
    polite = ["thank you", "thanks", "bye", "goodbye"]
    if msg in polite:
        return "You're welcome! Feel free to ask any medical questions."

    # --- Process medical questions through RAG ---
    try:
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "I'm not sure about that. Please consult a healthcare professional.")
    except Exception as e:
        print("Error during RAG processing:", e)
        answer = "Sorry, I couldn't process your question. Please try again."

    print("Response:", answer)
    return answer


# -----------------------
# Run the app
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
