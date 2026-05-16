"""
Assignment 6: Alien Language Translator
Few-Shot + Chain of Thought - Decode alien messages using examples and reasoning

Your mission: First contact! Decode alien communications using pattern
recognition and logical deduction!
"""

"""
Assignment 6: Alien Language Translator
Few-Shot + Chain of Thought - Decode alien messages using examples and reasoning

Your mission: First contact! Decode alien communications using pattern
recognition and logical deduction!
"""

import os
from typing import List
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

@dataclass
class Translation:
    alien_text: str
    human_text: str
    confidence: float
    reasoning_steps: List[str]
    cultural_notes: str

class AlienTranslator:
    """
    AI-powered alien language translator using few-shot examples and chain-of-thought reasoning.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.translation_examples = self._load_examples()
        self._setup_chains()

    def _load_examples(self) -> List[dict]:
        examples = [
            {
                "alien": "[translate:◈◈◈ ▲▲ ●]",
                "reasoning": "[translate:Step 1: ◈◈◈ appears to be quantity (3 symbols)]\n"
                             "[translate:Step 2: ▲▲ represents object type]\n"
                             "[translate:Step 3: ● is singular marker]\n"
                             "[translate:Step 4: Pattern suggests 'three ships approaching']",
                "translation": "[translate:Three ships approaching]",
                "pattern": "quantity-object-verb",
            },
            {
                "alien": "[translate:◆◆ ◆♦ ♦♦♦]",
                "reasoning": "[translate:Step 1: ◆◆ repeated indicates group size]\n"
                             "[translate:Step 2: ◆♦ ♦♦♦ symbols imply movement or travel]\n"
                             "[translate:Step 3: Overall meaning is 'two groups traveling']",
                "translation": "[translate:Two groups traveling]",
                "pattern": "group-movement",
            },
            {
                "alien": "[translate:▼▼ ●● ▲▲▲]",
                "reasoning": "[translate:Step 1: ▼▼ denotes warning or alert]\n"
                             "[translate:Step 2: ●● indicates plural subject]\n"
                             "[translate:Step 3: ▲▲▲ represents objects or entities involved]\n"
                             "[translate:Step 4: Interpretation: 'multiple alerts regarding entities']",
                "translation": "[translate:Multiple alerts regarding entities]",
                "pattern": "alert-subject-object",
            },
        ]
        return examples

    def _setup_chains(self):
        few_shot_examples = "\n\n".join([
            f"Alien: {ex['alien']}\nReasoning: {ex['reasoning']}\nTranslation: {ex['translation']}"
            for ex in self.translation_examples
        ])

        template = """
You are an expert alien language translator. Use the examples to decode the new message.

{few_shots}

Alien: {alien_message}
Reasoning:"""

        self.prompt_template = PromptTemplate(
            input_variables=["few_shots", "alien_message"],
            template=template
        )

    def translate(self, alien_message: str) -> Translation:
        prompt_text = self.prompt_template.format(
            few_shots="\n\n".join([
                f"Alien: {ex['alien']}\nReasoning: {ex['reasoning']}\nTranslation: {ex['translation']}"
                for ex in self.translation_examples
            ]),
            alien_message=alien_message
        )

        messages = [HumanMessage(content=prompt_text)]

        response = self.llm.invoke(messages)
        response_text = response.content

        lines = response_text.strip().split('\n')
        reasoning_steps = []
        translation_text = ""
        for line in lines:
            low_line = line.lower()
            if low_line.startswith("reasoning:"):
                reasoning_steps.append(line[len("Reasoning:"):].strip())
            elif low_line.startswith("translation:"):
                translation_text = line[len("Translation:"):].strip()
            else:
                reasoning_steps.append(line.strip())

        confidence = 0.75
        cultural_notes = ""

        return Translation(
            alien_text=alien_message,
            human_text=translation_text,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            cultural_notes=cultural_notes
        )


def test_translator():
    translator = AlienTranslator()

    test_messages = [
        "[translate:◈◈◈◈◈ ▲▲▲ ● ◆]",
        "[translate:♦♦ ◯◯◯ ▼ ★★★★]",
        "[translate:△△△ ◈ ■■ ◆◆◆]"
    ]

    print("👽 ALIEN LANGUAGE TRANSLATOR 👽")
    print("=" * 70)

    for msg in test_messages:
        result = translator.translate(msg)
        print(f"\nAlien: {msg}")
        print(f"Translation: {result.human_text}")
        print(f"Confidence: {result.confidence:.0%}")
        print("-" * 70)


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_translator()

