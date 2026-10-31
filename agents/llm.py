from functools import lru_cache

import os
from langchain_openrouter import ChatOpenRouter


@lru_cache(maxsize=None)
def get_llm(model_name):
    llm = ChatOpenRouter(
        model=model_name,
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        temperature=0
    )

    return llm.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True
    )