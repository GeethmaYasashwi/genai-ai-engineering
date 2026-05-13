import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import re

# Load environment variables from .env
load_dotenv()

class ViolationCategory(Enum):
    TEMPERATURE_CONTROL = "Food Temperature Control"
    PERSONAL_HYGIENE = "Personal Hygiene"
    PEST_CONTROL = "Pest Control"
    CROSS_CONTAMINATION = "Cross Contamination"
    FACILITY_MAINTENANCE = "Facility Maintenance"
    UNKNOWN = "Unknown"

class SeverityLevel(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class InspectionPriority(Enum):
    URGENT = "URGENT"
    HIGH = "HIGH"
    ROUTINE = "ROUTINE"
    LOW = "LOW"

@dataclass
class Violation:
    category: str
    description: str
    severity: str
    evidence: str
    confidence: float

@dataclass
class InspectionReport:
    restaurant_name: str
    overall_risk_score: int
    violations: List[Violation]
    inspection_priority: str
    recommended_actions: List[str]
    follow_up_required: bool

class FoodSafetyInspector:
    """
    AI-powered food safety analyzer using zero-shot structured prompting.
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.1):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.analysis_chain = None
        self.risk_chain = None
        self._setup_chains()

    def _setup_chains(self):
        analysis_template_str = (
        "You are a food safety inspector AI. Analyze the following text for health code violations.\n"
        "Your ONLY output should be a valid, minified JSON array (no comments, no explanations, no newlines):\n"
        '[{{"category": <string>, "description": <string>, "severity": <Critical|High|Medium|Low>, "evidence": <string>, "confidence": <float>}}]\n'
        "Categories: Temperature Control, Personal Hygiene, Pest Control, Cross Contamination, Facility Maintenance, Unknown.\n"
        "Quote evidence directly from the text. If ambiguity exists, assign lower confidence. If no violation detected, output [].\n"
        "Text: {review_text}\n"
        "Output:"
        )
        
        risk_template_str = (
        "Given these health code violations (as JSON), calculate an overall risk score from 0 to 100.\n"
        "Weights: Critical=40, High=30, Medium=20, Low=10. Add weights for each violation, clamp to 100 max.\n"
        "Assign inspection_priority: URGENT (>70), HIGH (51-70), ROUTINE (31-50), LOW (<=30).\n"
        "Output ONLY valid minified JSON:\n"
        '{{"risk_score": <int>, "inspection_priority": <string>}}\n'
        "Violations: {violations}\n"
        "Output:"
        )


        analysis_template = PromptTemplate.from_template(analysis_template_str)
        risk_template = PromptTemplate.from_template(risk_template_str)
        self.analysis_chain = analysis_template | self.llm
        self.risk_chain = risk_template | self.llm

    def detect_violations(self, text: str) -> List[Violation]:
        """
        Detect health violations from text input.
        Returns: List of Violation objects with evidence
        """
        try:
            raw_response = self.analysis_chain.invoke({"review_text": text})
            # Extract the text from the AIMessage object
            if hasattr(raw_response, "content"):
                response_content = raw_response.content
            else:
                response_content = str(raw_response)
            print("LLM Response Content:", response_content)
            if not response_content.strip():
                print("Empty response from LLM for text:", text)
                return []
            try:
                # Try direct JSON parse
                data = json.loads(response_content)
            except Exception:
                # Try to extract JSON array from a messy output
                matches = re.findall(r'\[.*?\]', response_content, flags=re.DOTALL)
                if matches:
                    data = json.loads(matches[0])
                else:
                    print("Error parsing LLM response, could not extract JSON:", response_content)
                    return []
            violations = []
            for v in data:
                violations.append(
                    Violation(
                        category=v.get("category", "Unknown"),
                        description=v.get("description", ""),
                        severity=v.get("severity", "Low"),
                        evidence=v.get("evidence", ""),
                        confidence=v.get("confidence", 0.0),
                    )
                )
            return violations
        except Exception as e:
            print(f"Error detecting violations: {e}")
            return []

    def calculate_risk_score(self, violations: List[Violation]) -> Tuple[int, str]:
        severity_weights = {
            "Critical": 40,
            "High": 30,
            "Medium": 20,
            "Low": 10,
        }
        risk_score = sum(severity_weights.get(v.severity, 10) for v in violations)
        risk_score = min(max(risk_score, 0), 100)
        if risk_score > 70:
            priority = InspectionPriority.URGENT.value
        elif risk_score > 50:
            priority = InspectionPriority.HIGH.value
        elif risk_score > 30:
            priority = InspectionPriority.ROUTINE.value
        else:
            priority = InspectionPriority.LOW.value
        return risk_score, priority

    def analyze_review(self, text: str, restaurant_name: str = "Unknown") -> InspectionReport:
        violations = self.detect_violations(text)
        risk_score, priority = self.calculate_risk_score(violations)
        recommendations = []
        follow_up = False

        if risk_score > 70:
            recommendations.append("Immediate inspection and corrective action required.")
            follow_up = True
        elif risk_score > 50:
            recommendations.append("Schedule inspection soon and monitor.")
            follow_up = True
        elif risk_score > 30:
            recommendations.append("Review practices and improve hygiene.")
        else:
            recommendations.append("Routine check recommended.")

        return InspectionReport(
            restaurant_name=restaurant_name,
            overall_risk_score=risk_score,
            violations=violations,
            inspection_priority=priority,
            recommended_actions=recommendations,
            follow_up_required=follow_up,
        )

    def batch_analyze(self, reviews: List[Dict[str, str]]) -> InspectionReport:
        all_violations = []
        for review in reviews:
            detected = self.detect_violations(review["text"])
            all_violations.extend(detected)
        # Remove duplicates based on category and description, favor higher confidence
        unique = {}
        for v in all_violations:
            key = (v.category, v.description)
            if key not in unique or unique[key].confidence < v.confidence:
                unique[key] = v
        filtered_violations = list(unique.values())
        risk_score, priority = self.calculate_risk_score(filtered_violations)

        recommendations = ["Aggregate findings from multiple review sources and prioritize accordingly."]
        follow_up = risk_score > 50
        return InspectionReport(
            restaurant_name="Aggregated",
            overall_risk_score=risk_score,
            violations=filtered_violations,
            inspection_priority=priority,
            recommended_actions=recommendations,
            follow_up_required=follow_up,
        )

    def filter_false_positives(self, violations: List[Violation]) -> List[Violation]:
        filtered = []
        for v in violations:
            if v.confidence < 0.5:
                continue
            if any(word in v.evidence.lower() for word in ["just kidding", "sarcasm", "not true"]):
                continue
            filtered.append(v)
        return filtered

def test_inspector():
    inspector = FoodSafetyInspector()

    test_reviews = [
        {
            "restaurant": "Bob's Burgers",
            "text": "Great food but saw a mouse run across the dining room! Also, the chef wasn't wearing gloves while handling raw chicken.",
        },
        {
            "restaurant": "Pizza Palace",
            "text": "Just left and the bathroom had no soap, and I'm pretty sure that meat sitting on the counter wasn't refrigerated 😷",
        },
        {
            "restaurant": "Sushi Express",
            "text": "Love this place! Though it's weird they keep the raw fish next to the vegetables #sushitime #questionable",
        },
        {
            "restaurant": "Taco Town",
            "text": "Best tacos in town! Super clean kitchen, staff always wears hairnets, everything looks fresh!",
        },
        {
            "restaurant": "Burger Barn",
            "text": "The cockroach in my salad added extra protein! Just kidding, but seriously the place needs cleaning.",
        },
    ]

    print("🍽️ FOOD SAFETY INSPECTION SYSTEM 🍽️\n")
    print("=" * 70)

    for review_data in test_reviews:
        print(f"\n🏪 Restaurant: {review_data['restaurant']}")
        print(f"📝 Review: \"{review_data['text'][:100]}...\"")
        report = inspector.analyze_review(review_data["text"], review_data["restaurant"])
        print(f"\n📊 Inspection Report:")
        print(f"  Risk Score: {report.overall_risk_score}/100")
        print(f"  Priority: {report.inspection_priority}")
        print(f"  Violations Found: {len(report.violations)}")
        if report.violations:
            print("\n  Detected Violations:")
            for v in report.violations:
                print(f"    • [{v.severity}] {v.category}: {v.description}")
                print(f'      Evidence: "{v.evidence[:50]}..."')
                print(f"      Confidence: {v.confidence:.0%}")
        if report.recommended_actions:
            print("\n  Recommended Actions:")
            for action in report.recommended_actions:
                print(f"    ✓ {action}")
        print(f"\n  Follow-up Required: {'Yes' if report.follow_up_required else 'No'}")
        print("-" * 70)

    # Test batch analysis
    print("\n🔬 BATCH ANALYSIS TEST:")
    print("=" * 70)
    batch_reviews = [
        {"text": "Saw bugs in the kitchen!", "source": "Yelp"},
        {"text": "Food was cold and undercooked", "source": "Google"},
        {"text": "Staff not wearing hairnets", "source": "Twitter"},
    ]
    batch_report = inspector.batch_analyze(batch_reviews)
    print(f"Aggregate Risk Score: {batch_report.overall_risk_score}/100")
    print(f"Total Violations: {len(batch_report.violations)}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    test_inspector()
