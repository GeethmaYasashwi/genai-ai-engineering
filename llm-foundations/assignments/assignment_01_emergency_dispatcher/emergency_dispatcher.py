import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# =========================
# DATA MODELS
# =========================

@dataclass
class DreamSymbol:
    symbol: str
    meaning: str
    frequency: int = 1
    significance: float = 0.5


@dataclass
class DreamAnalysis:
    symbols: List[DreamSymbol]
    emotions: List[str]
    themes: List[str]
    lucidity_score: float
    psychological_insights: str
    recurring_patterns: List[str]
    dream_type: str


# =========================
# DREAM ANALYZER
# =========================

class DreamAnalyzer:
    def __init__(self, model="gpt-4o-mini", temperature=0.2):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self._build_chains()

    def _build_chains(self):

        # -------- SYMBOLS --------
        symbol_prompt = PromptTemplate.from_template(
            """
            Extract dream symbols and meanings.

            Return ONLY JSON array:
            [
              {{
                "symbol": "...",
                "meaning": "...",
                "frequency": 1,
                "significance": 0.5
              }}
            ]

            Dream:
            {dream_text}
            """
        )

        # -------- EMOTIONS --------
        emotion_prompt = PromptTemplate.from_template(
            """
            Extract emotions from the dream.

            Return ONLY JSON:
            {{
              "emotions": ["fear", "joy"],
              "intensity": 0-10
            }}

            Dream:
            {dream_text}
            """
        )

        # -------- INSIGHTS --------
        insight_prompt = PromptTemplate.from_template(
            """
            Analyze the dream psychologically.

            Return ONLY JSON:
            {{
              "lucidity_score": 0-10,
              "themes": ["falling", "flying"],
              "recurring_patterns": ["..."],
              "psychological_insights": "..."
            }}

            Dream:
            {dream_text}

            Symbols:
            {symbols}

            Emotions:
            {emotions}
            """
        )

        # =========================
        # LCEL CHAINS
        # =========================
        self.symbol_chain = symbol_prompt | self.llm | JsonOutputParser()
        self.emotion_chain = emotion_prompt | self.llm | JsonOutputParser()
        self.insight_chain = insight_prompt | self.llm | JsonOutputParser()

    # =========================
    # CORE FUNCTIONS
    # =========================

    def extract_symbols(self, dream_text: str) -> List[DreamSymbol]:
        try:
            result = self.symbol_chain.invoke({"dream_text": dream_text})
            return [DreamSymbol(**s) for s in result]
        except Exception as e:
            print("Symbol extraction error:", e)
            return []

    def analyze_emotions(self, dream_text: str) -> Tuple[List[str], float]:
        try:
            result = self.emotion_chain.invoke({"dream_text": dream_text})
            return result.get("emotions", []), result.get("intensity", 5.0)
        except Exception as e:
            print("Emotion analysis error:", e)
            return [], 5.0

    def generate_insights(self, dream_text: str, symbols, emotions):
        try:
            result = self.insight_chain.invoke({
                "dream_text": dream_text,
                "symbols": json.dumps([asdict(s) for s in symbols]),
                "emotions": json.dumps(emotions)
            })
            return result
        except Exception as e:
            print("Insight generation error:", e)
            return {
                "lucidity_score": 0.0,
                "themes": [],
                "recurring_patterns": [],
                "psychological_insights": ""
            }

    def analyze_dream(self, dream_text: str) -> DreamAnalysis:
        print("Analyzing dream...")

        symbols = self.extract_symbols(dream_text)
        emotions, _ = self.analyze_emotions(dream_text)
        insights = self.generate_insights(dream_text, symbols, emotions)

        result = DreamAnalysis(
            symbols=symbols,
            emotions=emotions,
            themes=insights.get("themes", []),
            lucidity_score=insights.get("lucidity_score", 0.0),
            psychological_insights=insights.get("psychological_insights", ""),
            recurring_patterns=insights.get("recurring_patterns", []),
            dream_type="lucid" if insights.get("lucidity_score", 0) >= 7 else "normal"
        )

        return result

    def compare_dreams(self, d1: str, d2: str) -> Dict:
        a1 = self.analyze_dream(d1)
        a2 = self.analyze_dream(d2)

        shared_symbols = list(set(s.symbol for s in a1.symbols) &
                              set(s.symbol for s in a2.symbols))

        shared_themes = list(set(a1.themes) & set(a2.themes))

        return {
            "similarity_score": (
                len(shared_symbols) * 0.4 +
                len(shared_themes) * 0.6
            ),
            "shared_symbols": shared_symbols,
            "shared_themes": shared_themes
        }


if __name__ == "__main__":
    print("SCRIPT STARTED")

    analyzer = DreamAnalyzer()

    dream_text = "I was flying over a dark ocean while being chased by a shadow"

    result = analyzer.analyze_dream(dream_text)

    print("\nFINAL OUTPUT:\n")
    print(result)