from typing import List

from langchain.callbacks.base import BaseCallbackHandler
from langchain_community import chat_models
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import RagConfig


import inspect

def get_llm_model(config: RagConfig, callbacks: List[BaseCallbackHandler] = []):
    if config.llm.source == "ChatGoogleGenerativeAI":
        llm_spec = ChatGoogleGenerativeAI
    else:
        from langchain_community import chat_models
        llm_spec = getattr(chat_models, config.llm.source)

    # Obtenir la liste des paramètres du constructeur
    valid_params = inspect.signature(llm_spec).parameters

    kwargs = {
        key: value
        for key, value in config.llm.source_config.items()
        if key in valid_params
    }
    kwargs["streaming"] = True
    kwargs["callbacks"] = callbacks

    return llm_spec(**kwargs)
