"""
Assignment 10: AI Dungeon Master
The Ultimate Challenge - Master all prompting techniques to run a D&D game

Your mission: Become the ultimate AI Dungeon Master by seamlessly combining
all prompting techniques to create epic adventures!
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class QuestType(Enum):
    RESCUE = "rescue"
    FETCH = "fetch"
    INVESTIGATE = "investigate"
    COMBAT = "combat"
    DIPLOMACY = "diplomacy"
    EXPLORATION = "exploration"

@dataclass
class Character:
    name: str
    class_type: str
    level: int
    hit_points: int
    abilities: List[str]
    inventory: List[str]
    personality: str

@dataclass
class NPC:
    name: str
    role: str
    personality: str
    motivation: str
    dialogue_style: str
    secrets: List[str]

@dataclass
class Quest:
    title: str
    description: str
    objectives: List[str]
    rewards: List[str]
    difficulty: int
    quest_type: str

@dataclass
class CombatState:
    participants: List[Character]
    turn_order: List[str]
    environment: str
    special_conditions: List[str]

@dataclass
class WorldState:
    location: str
    time_of_day: str
    weather: str
    active_quests: List[Quest]
    npcs_present: List[NPC]
    recent_events: List[str]
    player_reputation: Dict[str, int]

class DungeonMasterAI:
    """
    AI Dungeon Master using all prompting techniques seamlessly.
    The ultimate test of prompting mastery!
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.story_generator = None  # Zero-shot
        self.npc_manager = None  # Few-shot
        self.combat_resolver = None  # CoT
        self.world_tracker = None  # Combined
        self.world_state = WorldState(
            location="Tavern",
            time_of_day="Evening",
            weather="Clear",
            active_quests=[],
            npcs_present=[],
            recent_events=[],
            player_reputation={},
        )
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Set up all chains for different DM tasks.
        Create:
        1. Zero-shot story generator for creative scenarios
        2. Few-shot NPC manager with personality examples
        3. CoT combat resolver for rule calculations
        4. Combined world tracker for state management
        """

        # Zero-shot for creative story generation
        self.story_generator = PromptTemplate.from_template(
            """Create an engaging D&D scenario.

Setting: {setting}
Plot hooks: {hooks}
Atmosphere: {atmosphere}
Player Agency: {agency}

Context: {context}
Player Action: {action}

Narration:"""
        )

        # Few-shot for NPC personalities
        npc_examples = [
            {
                "npc_type": "Gruff Innkeeper",
                "dialogue": "'Ale's two copper, room's a silver. No trouble or you're out.'",
                "personality": "Direct, no-nonsense, secretly kind",
                "quirk": "Always cleaning the same glass",
            },
            {
                "npc_type": "Mysterious Sage",
                "dialogue": "'The answer you seek lies not in what is seen, but what is hidden...'",
                "personality": "Cryptic, wise, slightly mad",
                "quirk": "Speaks in riddles and rhymes",
            },
            {
                "npc_type": "Nervous Merchant",
                "dialogue": "'Please... just don't hurt me. The goblins took everything!'",
                "personality": "Anxious, desperate, talkative",
                "quirk": "Fidgets with coin pouch",
            }
        ]
        self.npc_manager = FewShotPromptTemplate(
            examples=npc_examples,
            example_prompt=PromptTemplate.from_template(
                "NPC Type: {npc_type}\nPersonality: {personality}\nQuirk: {quirk}\nDialogue: {dialogue}\n"
            ),
            prefix="You are roleplaying an NPC in a D&D tavern. Answer as the NPC based on personality and player interaction.",
            suffix="NPC Type: {npc_type}\nPersonality: {personality}\nQuirk: {quirk}\nPlayer says: {player_input}\nNPC Response:",
            input_variables=["npc_type", "personality", "quirk", "player_input"],
        )

        # CoT for combat calculations
        self.combat_resolver = PromptTemplate.from_template(
            """Resolve this D&D combat action step by step.

Action: {action}
Attacker Stats: {stats}
Target Stats: {target}
Environment: {environment}

Let's calculate the outcome step by step:
Step 1: Check attack roll...
Step 2: Compare to target AC...
Step 3: Calculate damage...
Step 4: Adjust HP, describe effects.
End with: Did attack hit? Damage dealt. Special effects.
Combat Result:"""
        )

        # Combined approach for world state tracking
        self.world_tracker = PromptTemplate.from_template(
            """Update the world state based on player actions.

Track consequences with logical steps (CoT).
Generate reactions creatively (Zero-shot).
Maintain character consistency (Few-shot).

Current State: {current_state}
Player Actions: {actions}
Time Passed: {time}

Updated State:"""
        )

    def generate_quest(self, quest_type: QuestType, party_level: int) -> Quest:
        """
        TODO #2: Generate a quest using zero-shot creativity.
        Create unique, engaging quests without examples.
        """
        setting = random.choice(["Ancient ruins", "Haunted forest", "Bustling city", "Remote village"])
        hooks = random.choice([
            "A noble offers a reward to find a lost heirloom.",
            "Goblins threaten local merchants.",
            "A mysterious prophecy unfolds.",
            "Bandits kidnap a mayor's child.",
        ])
        atmosphere = random.choice([
            "Tense and mysterious", "Bright and lively", "Dark and ominous", "Eerie and foggy"
        ])
        agency = "Players can negotiate, fight, or explore multiple solutions."
        context = f"Quest Type: {quest_type.value}, Party Level: {party_level}"

        prompt = self.story_generator.format(
            setting=setting,
            hooks=hooks,
            atmosphere=atmosphere,
            agency=agency,
            context=context,
            action="The adventurers accept the quest."
        )
        response = self.llm.invoke(prompt).content

        lines = response.splitlines()
        title = lines[0] if lines else f"{quest_type.value.title()} Quest"
        description = " ".join(lines[1:3]) if len(lines) > 2 else response.strip()
        objectives = ["Complete the main objective", "Survive the challenge", "Discover secret"]
        rewards = ["Gold", "Magical item", "Favor from locals"]
        return Quest(
            title=title,
            description=description,
            objectives=objectives,
            rewards=rewards,
            difficulty=party_level,
            quest_type=quest_type.value,
        )

    def roleplay_npc(self, npc: NPC, player_input: str, context: Dict[str, any]) -> str:
        """
        TODO #3: Roleplay NPC using few-shot personality examples.
        Match personality patterns from examples.
        """
        prompt = self.npc_manager.format(
            npc_type=npc.role,
            personality=npc.personality,
            quirk=npc.dialogue_style,
            player_input=player_input
        )
        response = self.llm.invoke(prompt).content
        return response.strip()

    def resolve_combat(
        self,
        action: str,
        attacker: Character,
        target: Character,
        combat_state: CombatState,
    ) -> Dict[str, any]:
        """
        TODO #4: Resolve combat using CoT for rule calculations.
        Step-by-step D&D combat resolution.
        """
        stats = str(asdict(attacker))
        target_stats = str(asdict(target))
        env = combat_state.environment

        prompt = self.combat_resolver.format(
            action=action,
            stats=stats,
            target=target_stats,
            environment=env
        )
        response = self.llm.invoke(prompt).content
        # Simple parsing for demo
        hit = 'hit' in response.lower()
        damage = 0
        import re
        match = re.search(r'damage.*?(\\d+)', response.lower())
        if match:
            damage = int(match.group(1))
        special = []
        if "special" in response.lower():
            special.append("Special effect triggered")
        description = response.strip()
        return {"hit": hit, "damage": damage, "description": description, "special_effects": special}

    def narrate_scene(
        self, action: str, world_state: WorldState, characters: List[Character]
    ) -> str:
        """
        TODO #5: Narrate scene using zero-shot creativity.
        Generate atmospheric, engaging descriptions.
        """
        context = f"Location: {world_state.location}, Weather: {world_state.weather}, Time: {world_state.time_of_day}"
        pc_names = ', '.join([c.name for c in characters])
        prompt = self.story_generator.format(
            setting=world_state.location,
            hooks=f"Party: {pc_names}",
            atmosphere=world_state.weather,
            agency="Players may explore, react, or interact.",
            context=context,
            action=action
        )
        response = self.llm.invoke(prompt).content
        return response.strip()

    def update_world(self, actions: List[str], time_passed: str) -> WorldState:
        """
        TODO #6: Update world state using ALL techniques.
        Orchestrate all methods for comprehensive world management.
        """
        current_state = str(asdict(self.world_state))
        prompt = self.world_tracker.format(
            current_state=current_state,
            actions=", ".join(actions),
            time=time_passed
        )
        response = self.llm.invoke(prompt).content
        # For demo, just log recent events and update time_of_day simply
        self.world_state.recent_events.extend(actions)
        if "morning" in time_passed.lower():
            self.world_state.time_of_day = "Morning"
        elif "afternoon" in time_passed.lower():
            self.world_state.time_of_day = "Afternoon"
        elif "night" in time_passed.lower():
            self.world_state.time_of_day = "Night"
        else:
            self.world_state.time_of_day = "Evening"
        return self.world_state

    def run_session(
        self, player_actions: List[str], party: List[Character]
    ) -> Dict[str, any]:
        """
        TODO #7: Run a complete game session using all techniques.
        The ultimate test - seamlessly combine everything!
        """
        session_log = {
            "narration": [],
            "npc_interactions": [],
            "combat_results": [],
            "quest_updates": [],
            "world_changes": [],
        }
        for i, act in enumerate(player_actions):
            narration = self.narrate_scene(act, self.world_state, party)
            session_log["narration"].append(narration)
            for npc in self.world_state.npcs_present:
                npc_resp = self.roleplay_npc(npc, act, {"location": self.world_state.location})
                session_log["npc_interactions"].append(npc_resp)
            if "attack" in act.lower() or "fight" in act.lower():
                combat = CombatState(
                    participants=party,
                    turn_order=[p.name for p in party],
                    environment=self.world_state.location,
                    special_conditions=["Foggy" if "fog" in self.world_state.weather else ""],
                )
                cr = self.resolve_combat(act, party[0], party[-1], combat)
                session_log["combat_results"].append(cr)
            quest = self.generate_quest(random.choice(list(QuestType)), random.choice([c.level for c in party]))
            session_log["quest_updates"].append(asdict(quest))
            self.update_world([act], "1 hour")
            session_log["world_changes"].append(str(asdict(self.world_state)))
        return session_log

def test_dungeon_master():
    """Test the AI Dungeon Master with a mini adventure."""
    dm = DungeonMasterAI()
    # Create test party
    test_party = [
        Character(
            name="Aldric",
            class_type="Fighter",
            level=3,
            hit_points=28,
            abilities=["Second Wind", "Action Surge"],
            inventory=["Longsword", "Shield", "Healing Potion"],
            personality="Brave but reckless",
        ),
        Character(
            name="Lyra",
            class_type="Wizard",
            level=3,
            hit_points=18,
            abilities=["Fireball", "Shield", "Detect Magic"],
            inventory=["Spellbook", "Crystal Orb", "Scrolls"],
            personality="Cautious and analytical",
        ),
    ]
    # Create test NPCs
    test_npcs = [
        NPC(
            name="Gareth",
            role="Tavern Keeper",
            personality="Gruff but kind",
            motivation="Keep tavern safe",
            dialogue_style="Direct and practical",
            secrets=["Former adventurer", "Has a treasure map"],
        ),
        NPC(
            name="Lady Morwyn",
            role="Noble Patron",
            personality="Aristocratic and mysterious",
            motivation="Find ancient artifact",
            dialogue_style="Formal and cryptic",
            secrets=["Is actually a dragon", "Knows about the prophecy"],
        ),
    ]
    print("🎲 AI DUNGEON MASTER 🎲")
    print("=" * 70)
    print("Welcome to the Realm of Aethermoor!")
    print("-" * 70)
    # Test quest generation (Zero-shot)
    print("\n📜 QUEST GENERATION (Zero-shot):")
    quest = dm.generate_quest(QuestType.RESCUE, party_level=3)
    print(f"Quest: {quest.title}")
    print(f"Description: {quest.description}")
    print(
        f"Objectives: {', '.join(quest.objectives[:2]) if quest.objectives else 'None'}"
    )
    # Test NPC roleplay (Few-shot)
    print("\n🗣️ NPC INTERACTION (Few-shot):")
    player_input = "We're looking for adventure and gold!"
    for npc in test_npcs[:1]:
        response = dm.roleplay_npc(npc, player_input, {"location": "tavern"})
        print(f'{npc.name}: "{response}"')
    # Test combat resolution (CoT)
    print("\n⚔️ COMBAT RESOLUTION (Chain of Thought):")
    combat = CombatState(
        participants=test_party,
        turn_order=[p.name for p in test_party],
        environment="Dark forest clearing",
        special_conditions=["Fog - disadvantage on ranged attacks"],
    )
    combat_result = dm.resolve_combat(
        "Aldric attacks the goblin with his longsword",
        test_party[0],
        Character("Goblin", "Monster", 1, 7, ["Sneak"], ["Dagger"], "Cowardly"),
        combat,
    )
    print(f"Action: Aldric attacks")
    print(f"Result: {'Hit!' if combat_result.get('hit') else 'Miss!'}")
    print(f"Damage: {combat_result.get('damage', 0)}")
    # Test scene narration (Zero-shot)
    print("\n🎭 SCENE NARRATION (Zero-shot):")
    narration = dm.narrate_scene(
        "The party enters the ancient ruins", dm.world_state, test_party
    )
    print(
        f"DM: {narration[:200]}..." if narration else "DM: [Scene description pending]"
    )
    # Test world state update (All techniques)
    print("\n🌍 WORLD STATE UPDATE (All Techniques):")
    player_actions = [
        "Defeated the goblin raiders",
        "Rescued the merchant",
        "Found mysterious artifact",
    ]
    updated_state = dm.update_world(player_actions, "2 hours")
    print(f"Location: {updated_state.location}")
    print(f"Time: {updated_state.time_of_day}")
    print(f"Recent Events: {len(updated_state.recent_events)} recorded")
    # Run mini session
    print("\n🎮 MINI SESSION (All Techniques Combined):")
    print("=" * 70)
    session_actions = [
        "We investigate the strange noises from the cellar",
        "I cast Detect Magic on the mysterious door",
        "We try to negotiate with the creature",
    ]
    session = dm.run_session(session_actions, test_party)
    if session.get("narration"):
        print("Session Highlights:")
        for event in session["narration"][:3]:
            print(f"  • {event}")
    print("\n🏆 Adventure Continues...")
    print("The AI Dungeon Master awaits your next move!")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
    else:
        test_dungeon_master()
