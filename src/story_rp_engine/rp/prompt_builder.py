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
    greeting: Optional[str] = None,
    user_persona: str = "",
) -> str:
    """Builds the roleplay system prompt from a character card, optional lore, the user's persona and the opening greeting."""
    char_name = card.name

    def fill(text: Optional[str]) -> str:
        return replace_macros(text, char_name, user_name).strip()

    intro = (
        f"You are {char_name} in an ongoing roleplay with {user_name}. Write {char_name}'s dialogue, actions "
        f"and thoughts, plus narration and side characters as needed. Never write {user_name}'s words, actions or decisions."
    )
    sections = [intro]
    if card.system_prompt:
        sections.append(fill(card.system_prompt))

    sections.append(f"<character>\n{fill(card.description)}\n</character>")
    if card.personality:
        sections.append(f"<personality>\n{fill(card.personality)}\n</personality>")
    if card.scenario:
        sections.append(f"<scenario>\n{fill(card.scenario)}\n</scenario>")
    if user_persona:
        sections.append(f"<user_persona>\n{fill(user_persona)}\n</user_persona>")
    if active_lore:
        lore_snippets = "\n".join(f"- {entry.content}" for entry in active_lore)
        sections.append(f"<world_info>\n{lore_snippets}\n</world_info>")
    if card.mes_example:
        sections.append(
            f"<example_dialogue>\n{fill(card.mes_example)}\n</example_dialogue>\n"
            "The examples show voice and style only; they are not part of the story."
        )

    sections.append(
        "Guidelines:\n"
        "- Stay in character and consistent with established facts.\n"
        "- Reply in the language the conversation uses; usually 1-3 paragraphs.\n"
        f"- Move the scene forward: react, act, add small developments. Don't just mirror {user_name} "
        f"or end every reply with a question.\n"
        f"- Vary your wording; don't reuse phrases or openings from earlier replies.\n"
        f"- If {user_name} writes (OOC: ...), answer briefly out of character."
    )
    if greeting:
        sections.append(f"<opening_message>\n{fill(greeting)}\n</opening_message>\nYou already sent this opening message.")
    if card.post_history_instructions:
        sections.append(fill(card.post_history_instructions))

    return "\n\n".join(sections)
