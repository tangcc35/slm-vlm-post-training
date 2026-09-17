import re
from typing import Dict, List, Optional
from story_rp_engine.core.types import CharacterCardV2, ChatMessage, LorebookEntry


def replace_macros(text: Optional[str], char_name: str, user_name: str) -> str:
    """Replaces {{char}} and {{user}} macro placeholders (case-insensitive) with specified character and user names."""
    if not text:
        return ""
    text = re.sub(r"\{\{char\}\}", char_name, text, flags=re.IGNORECASE)
    text = re.sub(r"\{\{user\}\}", user_name, text, flags=re.IGNORECASE)
    return text


def build_rp_system_instruction(
    card: CharacterCardV2,
    active_lore: Optional[List[LorebookEntry]] = None,
    user_name: str = "User",
) -> str:
    """Builds a structured roleplay system instruction prompt from a character card and active lore entries."""
    data = card.data
    char_name = data.name

    sections = [
        f"You are roleplaying as {char_name}. Stay fully in character at all times. Do not break the fourth wall or speak for {user_name}.",
        f"### Character: {char_name}\n"
        f"- Description: {replace_macros(data.description, char_name, user_name)}\n"
        f"- Personality: {replace_macros(data.personality, char_name, user_name)}",
        f"### Scenario\n{replace_macros(data.scenario, char_name, user_name)}",
    ]

    if active_lore:
        lore_snippets = "\n".join([f"- {entry.content}" for entry in active_lore])
        sections.append(f"### World Information & Lore\n{lore_snippets}")

    if data.mes_example:
        sections.append(f"### Dialogue Examples\n{replace_macros(data.mes_example, char_name, user_name)}")

    if data.post_history_instructions:
        sections.append(f"### Additional Directives\n{replace_macros(data.post_history_instructions, char_name, user_name)}")

    return "\n\n".join(sections)


def assemble_history_with_steering(
    messages: List[ChatMessage],
    authors_note: Optional[str] = None,
    depth: int = 2,
    max_turns: int = 20,
) -> List[Dict[str, str]]:
    """Takes recent turns and injects an Author's Note steering directive at a given depth from the end."""
    truncated = messages[-max_turns:] if max_turns > 0 else []
    history = [{"role": msg.role, "content": msg.content} for msg in truncated]

    if authors_note and authors_note.strip():
        note_dict = {"role": "system", "content": authors_note.strip()}
        if depth <= 0 or depth >= len(history):
            history.append(note_dict)
        else:
            insert_idx = max(0, len(history) - depth)
            history.insert(insert_idx, note_dict)

    return history
