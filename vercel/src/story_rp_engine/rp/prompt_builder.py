import re
from typing import List, Optional
from story_rp_engine.core.types import CharacterCard, LorebookEntry


def replace_macros(text: Optional[str], char_name: str, user_name: str) -> str:
    """Replaces {{char}} and {{user}} macro placeholders (case-insensitive) with specified character and user names."""
    if not text:
        return ""
    text = re.sub(r"\{\{char\}\}", char_name, text, flags=re.IGNORECASE)
    text = re.sub(r"\{\{user\}\}", user_name, text, flags=re.IGNORECASE)
    return text


def build_rp_system_instruction(
    card: CharacterCard,
    active_lore: Optional[List[LorebookEntry]] = None,
    user_name: str = "User",
) -> str:
    """Builds a structured roleplay system instruction prompt from a character card and active lore entries."""
    char_name = card.name

    sections = [
        f"You are roleplaying as {char_name}. Stay fully in character at all times. Do not break the fourth wall or speak for {user_name}.",
        f"### Character: {char_name}\n"
        f"- Description: {replace_macros(card.description, char_name, user_name)}\n"
        f"- Personality: {replace_macros(card.personality, char_name, user_name)}",
        f"### Scenario\n{replace_macros(card.scenario, char_name, user_name)}",
    ]

    if active_lore:
        lore_snippets = "\n".join([f"- {entry.content}" for entry in active_lore])
        sections.append(f"### World Information & Lore\n{lore_snippets}")

    if card.mes_example:
        sections.append(f"### Dialogue Examples\n{replace_macros(card.mes_example, char_name, user_name)}")

    if card.post_history_instructions:
        sections.append(f"### Additional Directives\n{replace_macros(card.post_history_instructions, char_name, user_name)}")

    return "\n\n".join(sections)
