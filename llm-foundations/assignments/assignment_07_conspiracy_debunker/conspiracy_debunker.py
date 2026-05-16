"""
Assignment 7: Conspiracy Theory Debunker
Zero-Shot + Chain of Thought - Analyze and debunk misinformation

Your mission: Combat misinformation by analyzing conspiracy theories
with clear instructions and step-by-step logical reasoning!
"""

import os
from dataclasses import dataclass
from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import json

load_dotenv()

@dataclass
class DebunkAnalysis:
    conspiracy_text: str
    main_claims: List[str]
    logical_flaws: List[str]
    reasoning_chain: List[str]
    psychological_appeal: str
    debunking_summary: str
    reliable_sources: List[str]
    confidence_score: float

class ConspiracyDebunker:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.2)
        self._setup_chain()

    def _setup_chain(self):
        prompt_text = """
Analyze this conspiracy theory respectfully but critically.
Extract core claims, identify logical flaws through step-by-step reasoning,
outline the reasoning chain debunking the claims,
identify psychological appeal factors,
generate a respectful educational debunking summary,
and suggest reliable sources.

Return your answer as a valid JSON object matching this example exactly:

{{
  "main_claims": [
    "Claim 1 here",
    "Claim 2 here"
  ],
  "logical_flaws": [
    "Flaw explanation 1",
    "Flaw explanation 2"
  ],
  "reasoning_chain": [
    "Reasoning step 1",
    "Reasoning step 2"
  ],
  "psychological_appeal": "Brief explanation about psychological factors.",
  "debunking_summary": "Summary debunking the conspiracy theory.",
  "reliable_sources": [
    "Source 1",
    "Source 2"
  ],
  "confidence_score": 0.95
}}

Theory: {{conspiracy_text}}
"""
        self.prompt = PromptTemplate.from_template(prompt_text)
        self.analysis_chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def _extract_confidence(self, data) -> float:
        # Return confidence score if valid between 0 and 1,
        # else provide fallback heuristic.
        score = data.get("confidence_score")
        if isinstance(score, float) and 0.0 <= score <= 1.0:
            return score
        # Heuristic fallback: if main_claims and logical flaws present, assign medium confidence
        if data.get("main_claims") and data.get("logical_flaws"):
            return 0.75
        return 0.4  # Low confidence fallback

    def debunk(self, conspiracy_text: str) -> DebunkAnalysis:
        output = self.analysis_chain.invoke({"conspiracy_text": conspiracy_text})

        # output might be dict if LangChain parsed JSON; if not, try to parse string
        if isinstance(output, dict):
            data = output
        else:
            try:
                data = json.loads(output)
            except Exception:
                data = {}

        confidence = self._extract_confidence(data)

        return DebunkAnalysis(
            conspiracy_text=conspiracy_text,
            main_claims=data.get("main_claims", []),
            logical_flaws=data.get("logical_flaws", []),
            reasoning_chain=data.get("reasoning_chain", []),
            psychological_appeal=data.get("psychological_appeal", ""),
            debunking_summary=data.get("debunking_summary", ""),
            reliable_sources=data.get("reliable_sources", []),
            confidence_score=confidence,
        )

def test_debunker():
    debunker = ConspiracyDebunker()
    test_theories = [
        "Birds aren't real - they're government surveillance drones. Notice how they sit on power lines to recharge?",
        "The moon landing was filmed in a Hollywood studio. The flag waves despite no atmosphere!",
        "Chemtrails from planes are mind control chemicals. Normal contrails disappear quickly but these linger!",
    ]

    print("🤔 CONSPIRACY THEORY DEBUNKER 🤔")
    print("=" * 70)

    for theory in test_theories:
        result = debunker.debunk(theory)
        print(f'\nTheory: "{theory[:60]}..."')
        print(f"Main Claims: {len(result.main_claims)} identified")
        print(f"Logical Flaws: {len(result.logical_flaws)} found")
        print(f"Confidence: {result.confidence_score:.0%}")
        print("-" * 70)

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_debunker()
