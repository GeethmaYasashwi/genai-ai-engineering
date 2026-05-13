import os
import re
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class PrintStreamHandler(BaseCallbackHandler):
    """TODO: Print tokens to stdout as they arrive."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token, end="", flush=True)


class HaikuRephraser:
    def __init__(self):
        # TODO: Create a streaming LLM with PrintStreamHandler
        self.stream_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True, callbacks=[PrintStreamHandler()])
        # TODO: Create a non-streaming LLM for clean-up
        self.clean_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

        # Prompts
        stream_system = "You transform text into a 3-line haiku about a theme."
        stream_user = "Theme: {theme}\nText: {text}\nReturn only the haiku."
        clean_system = (
            "Ensure the haiku is crisp, natural, and fits 5-7-5 syllable spirit."
        )
        clean_user = "Polish this haiku while keeping its meaning:\n{draft}"

        # TODO: Build ChatPromptTemplates from the above strings
        self.stream_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(stream_system),
            HumanMessagePromptTemplate.from_template(stream_user),
        ])
        self.clean_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(clean_system),
            HumanMessagePromptTemplate.from_template(clean_user),
        ])

        # TODO: Build chains with StrOutputParser
        self.stream_chain = self.stream_prompt | self.stream_llm | StrOutputParser()
        self.clean_chain = self.clean_prompt | self.clean_llm | StrOutputParser()

    def rephrase(self, text: str, theme: str) -> str:
        """TODO: Stream a first pass, then run a clean-up pass and return final text."""
        draft = self.stream_chain.invoke({"text": text, "theme": theme})
        print()  
        final = self.clean_chain.invoke({"draft": draft})
        return final


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
        return
    r = HaikuRephraser()
    print("\n🌸 Haiku Rephraser — demo\n" + "-" * 40)
    result = r.rephrase("A quiet morning bus with foggy windows.", theme="dawn")
    print("\nPolished:\n" + result)


if __name__ == "__main__":
    _demo()
