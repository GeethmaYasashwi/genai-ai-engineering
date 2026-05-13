import os
import re
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class TransitExplainer:
    def __init__(self):
        # Create two LLMs: "calm" (low temperature) and "creative" (higher temperature/top_p)
        self.calm_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, top_p=1.0)
        self.creative_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8, top_p=0.9)

        # Build a role-aware prompt with {line_name} and {status_text}
        system_prompt = (
            "You rewrite internal operations notes into concise rider guidance "
            "with exactly two bullets: 1) Plain-language cause 2) What riders "
            "should do now. Keep it friendly and clear."
        )
        user_prompt = "Line: {line_name}\nStatus: {status_text}\nReturn only 2 bullets."

        # Create ChatPromptTemplate using the above strings
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt),
        ])

        # Create two chains with StrOutputParser
        self.calm_chain = self.prompt | self.calm_llm | StrOutputParser()
        self.creative_chain = self.prompt | self.creative_llm | StrOutputParser()

    def explain(self, line_name: str, status_text: str) -> str:
        """
        Invoke both chains and return the calm version.
        Optionally print the creative variant to compare tone.
        """
        calm = self.calm_chain.invoke({"line_name": line_name, "status_text": status_text})
        creative = self.creative_chain.invoke({"line_name": line_name, "status_text": status_text})
        print("\nCreative version for comparison:\n" + creative + "\n")
        return calm

def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    else:
        explainer = TransitExplainer()
        samples = [
            ("Green Line", "Signal failure near Station X causing cascading delays."),
            ("Red Line", "Unplanned track inspection between A and B, single-tracking in effect."),
        ]
        print("\n🚌 Transit Delay Explainer — demo\n" + "-" * 48)
        for line, status in samples:
            print(f"\nLine: {line}\nStatus: {status}")
            print(explainer.explain(line, status))

if __name__ == "__main__":
    _demo()
