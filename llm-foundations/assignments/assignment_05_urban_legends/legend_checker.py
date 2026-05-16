import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


class MythCategory(Enum):
    SUPERNATURAL = "supernatural"
    CONSPIRACY = "conspiracy"
    MEDICAL = "medical_health"
    TECHNOLOGY = "technology"
    HISTORICAL = "historical"
    SOCIAL = "social_phenomena"
    CREATURE = "cryptid_creature"


class LogicalFallacy(Enum):
    AD_HOMINEM = "ad_hominem"
    STRAW_MAN = "straw_man"
    FALSE_CAUSE = "false_cause"
    SLIPPERY_SLOPE = "slippery_slope"
    APPEAL_TO_AUTHORITY = "appeal_to_authority"
    CIRCULAR_REASONING = "circular_reasoning"
    HASTY_GENERALIZATION = "hasty_generalization"


@dataclass
class Claim:
    """Individual claim extracted from legend"""

    text: str
    testable: bool
    evidence_required: str
    confidence: float


@dataclass
class MythAnalysis:
    """Complete urban legend analysis"""

    original_text: str
    category: str
    claims: List[Claim]
    logical_fallacies: List[str]
    truth_rating: float  # 0 (false) to 1 (true)
    believability_score: float  # How convincing it sounds
    debunking_explanation: str
    similar_myths: List[str]
    origin_theory: str


class UrbanLegendChecker:
    """
    AI-powered urban legend analyzer combining zero-shot and few-shot prompting.
    Uses the right technique for each analysis task.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        """
        Initialize the legend checker.

        Args:
            model_name: The LLM model to use
            temperature: Controls randomness in responses
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.claim_extractor = None  # Zero-shot
        self.myth_classifier = None  # Few-shot
        self.fallacy_detector = None  # Combined
        self.debunker = None  # Zero-shot
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Set up different chains using appropriate prompting methods.

        Create:
        1. claim_extractor: Zero-shot for extracting claims
        2. myth_classifier: Few-shot for categorizing myths
        3. fallacy_detector: Combined approach for fallacies
        4. debunker: Zero-shot for generating explanations
        """

        # TODO: Zero-shot chain for claim extraction
        claim_template = PromptTemplate.from_template(
            """Extract all testable claims from this urban legend.

What constitutes a claim: a statement that can be verified or falsified.
How to determine if testable: the claim should be specific and able to be disproved or confirmed.
Evidence needed: factual data or scientific studies that support or refute the claim.
Output format: JSON array with fields text (claim), testable (bool), evidence_required (text), confidence (0-1).

Text: {legend_text}

Claims:"""
        )

        # TODO: Few-shot chain for myth classification
        classification_examples = [
            {
                "legend": "Alligators live in NYC sewers after being flushed as pets.",
                "category": "cryptid_creature",
                "reasoning": "Involves cryptid/hidden creature in urban setting",
            },
            {
                "legend": "Cell phones at gas stations can cause explosions.",
                "category": "technology",
                "reasoning": "Technology-related safety myth",
            },
            {
                "legend": "The government is tracking citizens through their smart devices.",
                "category": "conspiracy",
                "reasoning": "Paranoia about government surveillance",
            },
            {
                "legend": "Vaccines cause autism and other health problems.",
                "category": "medical_health",
                "reasoning": "Medical misinformation myth",
            },
            {
                "legend": "Ghosts haunt old houses and cemeteries.",
                "category": "supernatural",
                "reasoning": "Supernatural beings and hauntings",
            },
        ]

        classification_prompt = PromptTemplate.from_template(
            """Legend: {legend}
Category: {category}
Reasoning: {reasoning}"""
        )

        self.myth_classifier = FewShotPromptTemplate(
            examples=classification_examples,
            example_prompt=classification_prompt,
            input_variables=["legend"],
            prefix="Classify the urban legend into these categories: supernatural, conspiracy, medical_health, technology, historical, social_phenomena, cryptid_creature.\nProvide category and reasoning.\n",
            suffix="Legend: {legend}\nCategory:",
            example_separator="\n\n",
        )

        # TODO: Combined approach for fallacy detection
        fallacy_examples = [
            {
                "text": "Ad hominem fallacy: attacking the person, not the argument.",
                "fallacy": "ad_hominem",
            },
            {
                "text": "Straw man fallacy: misrepresenting argument to make it easier to attack.",
                "fallacy": "straw_man",
            },
            {"text": "False cause fallacy: assuming cause without proof.", "fallacy": "false_cause"},
            {
                "text": "Slippery slope fallacy: arguing that a minor step leads to major consequences without evidence.",
                "fallacy": "slippery_slope",
            },
        ]

        self.fallacy_detector_examples = fallacy_examples
        self.fallacy_detector_prefix = (
            "Detect logical fallacies in this urban legend. Explain briefly why. Use these examples as guides:\n"
        )
        self.fallacy_detector_suffix = "Legend: {legend}\nFallacies:"

        # TODO: Zero-shot chain for debunking explanations
        debunk_template = PromptTemplate.from_template(
            """Generate a clear, factual explanation debunking this myth.

Respectful tone. Scientific approach. Simple language. Address why people believe it.

Myth: {myth_text}
Claims: {claims}
Fallacies: {fallacies}

Debunking:"""
        )

        # Save prompt templates for use later
        self.claim_extractor = claim_template
        self.debunker = debunk_template

    def extract_claims_zero_shot(self, legend_text: str) -> List[Claim]:
        """
        TODO #2: Extract claims using zero-shot prompting.

        Args:
            legend_text: The urban legend text

        Returns:
            List of Claim objects
        """

        prompt_text = self.claim_extractor.format(legend_text=legend_text)
        response = self.llm.invoke(prompt_text)

        claims = []
        try:
            claims_json = json.loads(response)
            for claim_item in claims_json:
                claim = Claim(
                    text=claim_item.get("text", ""),
                    testable=claim_item.get("testable", False),
                    evidence_required=claim_item.get("evidence_required", ""),
                    confidence=claim_item.get("confidence", 0.0),
                )
                claims.append(claim)
        except Exception:
            # Fallback: no claims if JSON parsing fails
            pass

        return claims

    def classify_myth_few_shot(self, legend_text: str) -> Tuple[str, str]:
        """
        TODO #3: Classify myth type using few-shot examples.

        Args:
            legend_text: The urban legend text

        Returns:
            Tuple of (category, reasoning)
        """

        prompt_text = self.myth_classifier.format(legend=legend_text)
        response = self.llm.call_as_llm(prompt_text).strip()

        # Parse output: first line is category, rest reasoning
        lines = response.split("\n")
        category = lines[0].strip().lower() if lines else MythCategory.SUPERNATURAL.value
        reasoning = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        # Validate category
        valid_categories = {cat.value for cat in MythCategory}
        if category not in valid_categories:
            category = MythCategory.SUPERNATURAL.value

        return category, reasoning

    def detect_fallacies_combined(self, legend_text: str, claims: List[Claim]) -> List[str]:
        """
        TODO #4: Detect logical fallacies using combined approach.

        Args:
            legend_text: The urban legend text
            claims: Extracted claims

        Returns:
            List of detected fallacies with explanations
        """

        examples_text = "\n".join([f"{ex['text']} ({ex['fallacy']})" for ex in self.fallacy_detector_examples])
        prompt_text = (
            self.fallacy_detector_prefix
            + examples_text
            + "\n\n"
            + self.fallacy_detector_suffix.format(legend=legend_text)
        )
        response = self.llm.call_as_llm(prompt_text).strip()

        # Extract fallacies mentioned in response
        fallacies = []
        for fallacy in LogicalFallacy:
            if fallacy.value in response.lower():
                fallacies.append(fallacy.value)

        return list(set(fallacies))

    def calculate_believability(self, legend_text: str, claims: List[Claim], fallacies: List[str]) -> float:
        """
        TODO #5: Calculate how believable the myth sounds.

        Args:
            legend_text: The urban legend
            claims: Extracted claims
            fallacies: Detected fallacies

        Returns:
            Believability score 0-1
        """

        score = 0.5

        if fallacies:
            score -= 0.3  # Deduct if logical fallacies detected

        if claims:
            testable_count = sum(claim.testable for claim in claims)
            score += 0.1 * testable_count  # Increase slightly for testable claims

        return max(0.0, min(score, 1.0))

    def find_similar_myths(self, legend_text: str, category: str) -> List[str]:
        """
        TODO #6: Find similar myths using few-shot pattern matching.

        Args:
            legend_text: Current legend
            category: Myth category

        Returns:
            List of similar myth descriptions
        """

        similar_patterns = {
            "cryptid_creature": [
                "Alligators in NYC sewers",
                "Bigfoot sightings in the forest",
                "Loch Ness monster sightings",
            ],
            "technology": [
                "Cell phones cause explosions",
                "WiFi causes health problems",
                "Microwaves cause cancer",
            ],
            "conspiracy": [
                "Government mind control programs",
                "Secret societies control events",
                "Alien cover-ups by officials",
            ],
            "medical_health": [
                "Vaccines cause autism",
                "Homeopathy cures everything",
                "Detox diets cure disease",
            ],
            "supernatural": [
                "Ghosts haunt old houses",
                "Haunted cemeteries stories",
                "Vampires still exist",
            ],
            "historical": [
                "Napoleon was extremely short",
                "Columbus discovered America",
                "Einstein failed math",
            ],
            "social_phenomena": [
                "Mass hysteria outbreaks",
                "Urban legends spreading rapidly",
                "Viral social media hoaxes",
            ],
        }

        return similar_patterns.get(category, [])[:3]

    def analyze_legend(self, legend_text: str) -> MythAnalysis:
        """
        TODO #7: Complete analysis combining all methods.

        Args:
            legend_text: The urban legend to analyze

        Returns:
            Complete MythAnalysis object
        """

        claims = self.extract_claims_zero_shot(legend_text)
        category, _ = self.classify_myth_few_shot(legend_text)
        fallacies = self.detect_fallacies_combined(legend_text, claims)
        believability = self.calculate_believability(legend_text, claims, fallacies)

        # Rough truth rating: average claim confidence minus penalty for fallacies
        avg_confidence = (
            sum(claim.confidence for claim in claims) / len(claims) if claims else 0.0
        )
        truth_rating = avg_confidence - 0.2 * len(fallacies)
        truth_rating = max(0.0, min(truth_rating, 1.0))

        claims_json_str = json.dumps([asdict(c) for c in claims], indent=2)
        fallacies_text = ", ".join(fallacies) if fallacies else "None"

        debunking_prompt = self.debunker.format(
            myth_text=legend_text, claims=claims_json_str, fallacies=fallacies_text
        )
        debunking_explanation = self.llm.call_as_llm(debunking_prompt).strip()

        similar = self.find_similar_myths(legend_text, category)
        origin_theory = "Origin theory not implemented."

        return MythAnalysis(
            original_text=legend_text,
            category=category,
            claims=claims,
            logical_fallacies=fallacies,
            truth_rating=truth_rating,
            believability_score=believability,
            debunking_explanation=debunking_explanation,
            similar_myths=similar,
            origin_theory=origin_theory,
        )

    def adaptive_analysis(self, legend_text: str) -> Dict[str, any]:
        """
        TODO #8 (Bonus): Adaptively choose prompting method based on task.

        Args:
            legend_text: The legend to analyze

        Returns:
            Analysis with method choices explained
        """

        method_choices = {
            "claim_extraction": "zero-shot",
            "myth_classification": "few-shot",
            "fallacy_detection": "combined",
            "debunking": "zero-shot",
        }

        reasoning = (
            "Used zero-shot for novel tasks like claim extraction and debunking for flexibility. "
            "Few-shot used for classification to leverage patterns/examples. "
            "Combined approach picked for fallacy detection to cover known fallacies plus generalization."
        )

        analysis = self.analyze_legend(legend_text)

        confidence_scores = {
            "claim_extraction": 0.8,
            "myth_classification": 0.9,
            "fallacy_detection": 0.7,
            "debunking": 0.85,
        }

        return {
            "analysis": analysis,
            "method_choices": method_choices,
            "confidence_scores": confidence_scores,
            "reasoning": reasoning,
        }


def test_legend_checker():
    """Test the urban legend checker with various myths."""

    checker = UrbanLegendChecker()

    # Test legends of various types
    test_legends = [
        {
            "title": "The Vanishing Hitchhiker",
            "text": "A driver picks up a young woman hitchhiking on a dark road. She gives an address and sits silently in the back. When they arrive, she's vanished, leaving only a wet spot. The homeowner says she died in a car accident years ago on that very road.",
        },
        {
            "title": "5G Tower Mind Control",
            "text": "5G towers emit special frequencies that can control human thoughts and emotions. The government uses these towers to manipulate public opinion and behavior. People living near 5G towers report more headaches and mood changes, proving the effect.",
        },
        {
            "title": "The $250 Cookie Recipe",
            "text": "A woman at Neiman Marcus loved their cookies and asked for the recipe. The clerk said it would cost 'two-fifty' and she agreed. Her credit card was charged $250, not $2.50. In revenge, she's sharing the secret recipe with everyone.",
        },
        {
            "title": "Kidney Theft Ring",
            "text": "Business travelers are being drugged in hotel bars and waking up in bathtubs full of ice with their kidneys surgically removed. A note tells them to call 911 immediately. Hospitals confirm finding victims with professional surgical scars.",
        },
        {
            "title": "Pop Rocks and Soda",
            "text": "Mixing Pop Rocks candy with soda creates a chemical reaction that can cause your stomach to explode. A child actor died this way in the 1970s, which is why you never see them together in stores.",
        },
    ]

    print("🕵️ URBAN LEGEND FACT CHECKER 🕵️")
    print("=" * 70)

    for legend in test_legends:
        print(f"\n📚 Legend: {legend['title']}")
        print(f"📖 Story: \"{legend['text'][:80]}...\"")

        # Analyze the legend
        analysis = checker.analyze_legend(legend["text"])

        # Display results
        print(f"\n📊 Analysis Results:")
        print(f"  Category: {analysis.category}")
        print(f"  Truth Rating: {analysis.truth_rating:.0%}")
        print(f"  Believability: {analysis.believability_score:.0%}")

        if analysis.claims:
            print(f"\n  🎯 Claims Extracted ({len(analysis.claims)}):")
            for claim in analysis.claims[:2]:  # Show first 2
                print(f"    • {claim.text[:60]}...")
                print(f"      Testable: {'Yes' if claim.testable else 'No'}")

        if analysis.logical_fallacies:
            print(f"\n  ⚠️ Logical Fallacies Detected:")
            for fallacy in analysis.logical_fallacies[:2]:
                print(f"    • {fallacy}")

        if analysis.debunking_explanation:
            print(f"\n  📝 Debunking:")
            print(f"    {analysis.debunking_explanation[:150]}...")

        if analysis.similar_myths:
            print(f"\n  🔗 Similar Myths:")
            for myth in analysis.similar_myths[:2]:
                print(f"    • {myth}")

        print("-" * 70)

    # Test adaptive analysis
    print("\n🧠 ADAPTIVE ANALYSIS TEST:")
    print("=" * 70)

    complex_legend = "Ancient aliens built the pyramids using anti-gravity technology. The precise alignment with stars and mathematical perfection couldn't be achieved with primitive tools."

    adaptive_result = checker.adaptive_analysis(complex_legend)

    if adaptive_result.get("method_choices"):
        print("Method Selection:")
        for task, method in adaptive_result["method_choices"].items():
            print(f"  {task}: {method}")

    if adaptive_result.get("reasoning"):
        print(f"\nReasoning: {adaptive_result['reasoning']}")


if __name__ == "__main__":
    # Make sure to set OPENAI_API_KEY environment variable
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
    else:
        test_legend_checker()
