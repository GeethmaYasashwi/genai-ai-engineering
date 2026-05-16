"""
Assignment 8: Superhero Power Balancer
All Concepts Combined - Master all prompting techniques together
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv

load_dotenv()

class PowerType(Enum):
    PHYSICAL = "physical"
    ENERGY = "energy"
    MENTAL = "mental"
    REALITY = "reality"
    TECH = "technology"
    MAGIC = "magic"


@dataclass
class Hero:
    name: str
    abilities: List[str]
    power_type: str
    power_level: float
    weaknesses: List[str]
    synergies: List[str]


@dataclass
class BalanceReport:
    hero: Hero
    analysis_method: str  # Which prompting method was used
    power_rating: float
    balance_issues: List[str]
    suggested_changes: List[str]
    team_synergies: Dict[str, float]
    counter_picks: List[str]


class PowerBalancer:
    """
    AI-powered game balancer using all prompting techniques.
    Combines zero-shot, few-shot, and CoT for comprehensive analysis.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.4)
        self._setup_chains()

    def _setup_chains(self):
        """
        Set up prompts for each prompting technique.
        """

        # Zero-shot prompt template for ability analysis
        self.ability_template = PromptTemplate.from_template(
            """Analyze this superhero ability for game balance.

- Estimate the overall power level on a scale of 1-10.
- Identify any potential exploits or game-breaking aspects.
- Suggest counter-play or weaknesses.

Ability: {ability_description}

Analysis:"""
        )

        # Few-shot pattern matching for power type classification
        self.type_examples = [
            {
                "ability": "Super strength and invulnerability",
                "type": "physical",
                "reasoning": "Direct physical enhancement",
            },
            {
                "ability": "Mind reading and telepathy",
                "type": "mental",
                "reasoning": "Psychic mental powers",
            },
            {
                "ability": "Control over fire and flames",
                "type": "energy",
                "reasoning": "Elemental energy control",
            },
            {
                "ability": "Time manipulation and age acceleration",
                "type": "reality",
                "reasoning": "Manipulation of reality fabric",
            },
            {
                "ability": "Technological suit with AI assistance",
                "type": "technology",
                "reasoning": "Tech-based enhancement",
            },
            {
                "ability": "Casting spells and summoning demons",
                "type": "magic",
                "reasoning": "Arcane mystical powers",
            },
        ]

        # CoT prompt for interaction calculations
        self.interaction_template = PromptTemplate.from_template(
            """Calculate how these abilities interact in combat.

Ability 1: {ability1}
Ability 2: {ability2}

Let's think step by step about their interaction:"""
        )

    def analyze_hero_zero_shot(self, hero: Hero) -> Dict[str, any]:
        """
        Zero-shot analysis for novel or unique abilities.
        """
        prompt = self.ability_template.format(
            ability_description=", ".join(hero.abilities)
        )
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        result_text = response.content
        # Try parsing power level from response text with regex
        power_level_match = re.search(r"power level[: ]*([0-9.]+)", result_text, re.I)
        power_level = float(power_level_match.group(1)) if power_level_match else 5.0
        # Could extend to parse exploits and counters similarly
        return {"power_level": power_level, "exploits": [], "counters": []}

    def classify_power_few_shot(self, abilities: List[str]) -> str:
        """
        Few-shot classifier for power type pattern matching.
        """
        examples_text = ""
        for ex in self.type_examples:
            examples_text += f"""Ability: {ex['ability']}
Type: {ex['type']}
Reasoning: {ex['reasoning']}

"""
        abilities_text = ", ".join(abilities)
        prompt = f"""{examples_text}
Abilities: {abilities_text}
Determine the power type of these abilities based on the above examples.
Type:"""
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        predicted_type = response.content.strip().lower()
        # Validate predicted type and fallback
        for pt in PowerType:
            if pt.value == predicted_type:
                return pt.value
        return PowerType.PHYSICAL.value

    def calculate_synergy_cot(self, hero1: Hero, hero2: Hero) -> float:
        """
        Use Chain of Thought for complex interaction logic.
        """
        prompt = self.interaction_template.format(
            ability1=", ".join(hero1.abilities),
            ability2=", ".join(hero2.abilities),
        )
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        result_text = response.content
        # Example heuristic for synergy score parsing
        synergy_match = re.search(r"(\d{1,3})\s*%|(\d\.\d+)", result_text)
        if synergy_match:
            if synergy_match.group(1):
                score = int(synergy_match.group(1)) / 100
            elif synergy_match.group(2):
                score = float(synergy_match.group(2))
            else:
                score = 0.5
        else:
            score = 0.5  # default synergy score
        return score

    def detect_imbalance_combined(self, hero: Hero, meta: List[Hero]) -> BalanceReport:
        """
        Combine all techniques to analyze hero balance.
        """
        zero_shot_stats = self.analyze_hero_zero_shot(hero)
        p_type = self.classify_power_few_shot(hero.abilities)
        synergy_scores = {}
        for h in meta:
            if h.name != hero.name:
                synergy_scores[h.name] = self.calculate_synergy_cot(hero, h)
        issues = []

        if zero_shot_stats["power_level"] > 8:
            issues.append("Overpowered hero with high potential exploits.")
        if p_type == "magic" and "arcane" in " ".join(hero.abilities).lower():
            issues.append("Potential for game-breaking magic combos.")

        changes = []
        if "Overpowered hero" in issues:
            changes.append("Reduce hero power level or limit abilities.")
        if "magic" in issues:
            changes.append("Balance magic abilities to prevent instant wins.")

        balance_score = 10 - zero_shot_stats["power_level"]

        return BalanceReport(
            hero=hero,
            analysis_method="combined",
            power_rating=balance_score,
            balance_issues=issues,
            suggested_changes=changes,
            team_synergies=synergy_scores,
            counter_picks=["Anti-mind control hero", "Magic nullifier"],
        )

    def auto_balance(self, hero: Hero, target_power: float) -> Hero:
        """
        Auto-adjust hero attributes to match target power level.
        """
        # For demo, just adjust the power level attribute in a bounded way
        new_power_level = min(max(target_power, 1.0), 10.0)
        # Leave abilities as-is, but a real version can modify or nerf abilities here
        return Hero(
            name=hero.name,
            abilities=hero.abilities,
            power_type=hero.power_type,
            power_level=new_power_level,
            weaknesses=hero.weaknesses,
            synergies=hero.synergies,
        )


def test_balancer():
    balancer = PowerBalancer()

    test_heroes = [
        Hero(
            name="Chronos",
            abilities=["Time manipulation", "Temporal loops", "Age acceleration"],
            power_type="reality",
            power_level=9.0,
            weaknesses=[],
            synergies=[],
        ),
        Hero(
            name="Mindweaver",
            abilities=["Telepathy", "Illusion creation", "Memory manipulation"],
            power_type="mental",
            power_level=7.5,
            weaknesses=[],
            synergies=[],
        ),
        Hero(
            name="Quantum",
            abilities=["Teleportation", "Probability manipulation", "Phase shifting"],
            power_type="reality",
            power_level=8.5,
            weaknesses=[],
            synergies=[],
        ),
    ]

    print("⚡ SUPERHERO POWER BALANCER ⚡")
    print("=" * 70)

    for hero in test_heroes:
        print(f"\n🦸 Hero: {hero.name}")
        print(f"Abilities: {', '.join(hero.abilities)}")

        # Zero-shot analysis
        analysis = balancer.analyze_hero_zero_shot(hero)
        print(f"Power Level (Zero-shot): {analysis.get('power_level', 0):.1f}/10")

        # Few-shot classification
        power_type = balancer.classify_power_few_shot(hero.abilities)
        print(f"Classified Power Type: {power_type}")

        # CoT synergy with first hero if different
        if len(test_heroes) > 1 and hero.name != test_heroes[0].name:
            synergy = balancer.calculate_synergy_cot(hero, test_heroes[0])
            print(f"Synergy with {test_heroes[0].name}: {synergy:.0%}")

        print("-" * 70)

    # Balance analysis with all techniques
    print("\n🎯 BALANCE ANALYSIS (All Techniques):")
    print("=" * 70)

    report = balancer.detect_imbalance_combined(test_heroes[0], test_heroes)

    print(f"Hero: {report.hero.name}")
    print(f"Analysis Method: {report.analysis_method}")
    print(f"Power Rating: {report.power_rating:.1f}/10")
    if report.balance_issues:
        print("Balance Issues:")
        for issue in report.balance_issues:
            print(f"  ⚠️ {issue}")
    if report.suggested_changes:
        print("Suggested Changes:")
        for change in report.suggested_changes:
            print(f"  ✓ {change}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_balancer()
