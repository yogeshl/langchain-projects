from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

## Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="API Server"
)

# add_routes(
#     app,
#     ChatOpenAI(),
#     path="/openai"
# )

model = ChatOpenAI(model="gpt-3.5-turbo")
## ollama model
llm = Ollama(model="CodeExpert")

prompt1 = ChatPromptTemplate.from_template("Create a essay about {topic} within 100 words.")
prompt2 = ChatPromptTemplate.from_template("Create a nodejs program on {topic}.")


add_routes(
    app,
    prompt1|model,
    path="/essay"
)
add_routes(
    app,
    prompt2|model,
    path="/code"
)

if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)