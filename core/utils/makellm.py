
import getpass
import os


if not os.environ.get("LMSTUDIO_API_KEY"):
    os.environ["LMSTUDIO_API_KEY"] = getpass.getpass("Enter your LMSTUDIO_API_KEY: ")

def _make_llm(model, temp):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model = model,
        temperature= temp,
        base_url="http://localhost:1234/v1",
        api_key=os.environ.get("LMSTUDIO_API_KEY"),
    )
    return llm
