import re
from typing import List, Optional
from story_rp_engine.core.types import CharacterCard, GroupCard
from story_rp_engine.rp.prompt_builder import replace_macros


def group_agent_name(char_id: str) -> str:
    """ADK agent name for a group member; its events are authored under this name.

    Agent names must be identifiers. str.isidentifier accepts non-ASCII letters, so only non-word characters
    are replaced and CJK IDs stay distinct.
    """
    return "char_" + re.sub(r"\W", "_", char_id)


def _brief(text: str, limit: int = 300) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def build_group_system_instruction(
    card: CharacterCard,
    group: GroupCard,
    cards: List[CharacterCard],
    user_name: str = "User",
    greeting: Optional[str] = None,
) -> str:
    """Builds a group member's system prompt: its own card, with the group's scenario and the other members."""
    char_name = card.name

    def fill(text: Optional[str]) -> str:
        return replace_macros(text, char_name, user_name).strip()

    others = [c for c in cards if c.char_id != card.char_id]
    company = ", ".join([user_name] + [c.name for c in others])
    sections = [
        f"You are {char_name} in a group roleplay with {company}. Write only {char_name}'s dialogue, actions and "
        f"thoughts. Never write the words, actions or decisions of {user_name} or the other characters; they speak "
        "for themselves."
    ]
    if card.system_prompt:
        sections.append(fill(card.system_prompt))

    sections.append(f"<character>\n{fill(card.description)}\n</character>")
    if card.personality:
        sections.append(f"<personality>\n{fill(card.personality)}\n</personality>")
    if group.scenario:
        sections.append(f"<scenario>\n{fill(group.scenario)}\n</scenario>")
    if others:
        cast = "\n".join(
            f"- {c.name} (shown as [{group_agent_name(c.char_id)}]): "
            f"{_brief(replace_macros(c.description, c.name, user_name))}".rstrip()
            for c in others
        )
        sections.append(
            "<other_characters>\nThe conversation shows the other characters' lines as \"[name] said:\".\n"
            f"{cast}\n</other_characters>"
        )
    if card.mes_example:
        sections.append(
            f"<example_dialogue>\n{fill(card.mes_example)}\n</example_dialogue>\n"
            "The examples show voice and style only; they are not part of the story."
        )

    sections.append(
        "Guidelines:\n"
        "- Stay in character and consistent with established facts.\n"
        "- Reply in the language the conversation uses; usually 1-2 paragraphs, since the others reply too.\n"
        f"- React to what {user_name} and the others just said; don't repeat their lines or narrate their reactions.\n"
        "- Vary your wording; don't reuse phrases or openings from earlier replies.\n"
        f"- If {user_name} writes (OOC: ...), answer briefly out of character."
    )
    if greeting:
        sections.append(
            f"<opening_message>\n{fill(greeting)}\n</opening_message>\n"
            f"This message opened the scene before {user_name}'s first message."
        )
    if card.post_history_instructions:
        sections.append(fill(card.post_history_instructions))

    return "\n\n".join(sections)


def build_selector_instruction(group: GroupCard, cards: List[CharacterCard], user_name: str = "User") -> str:
    """Builds the speaker selector's system prompt: the members, the scene, and the JSON it must return."""
    members = "\n".join(
        f"- {c.char_id}: {c.name} (shown as [{group_agent_name(c.char_id)}]). "
        f"{_brief(replace_macros(c.description, c.name, user_name))}".rstrip()
        for c in cards
    )
    sections = [f"You decide who speaks next in a group roleplay between {user_name} and these characters:\n{members}"]
    if group.scenario:
        sections.append(f"Scenario:\n{replace_macros(group.scenario, group.name, user_name).strip()}")
    sections.append(
        f"Read the conversation and choose which characters reply to {user_name}'s latest message, in speaking "
        f"order. Put first any character {user_name} addresses. Include the characters who would naturally react "
        "and leave out those with nothing to add, but choose at least one.\n"
        'Reply with JSON only, using the ids above: {"speakers": ["<id>", ...]}'
    )
    return "\n\n".join(sections)


def group_summary_prompt(cards: List[CharacterCard]) -> str:
    """Compaction prompt for a group chat. ADK fills it with str.format, so braces in names are doubled."""
    cast = "\n".join(f"- {group_agent_name(c.char_id)}: {c.name}" for c in cards)
    cast = cast.replace("{", "{{").replace("}", "}}")
    return (
        "Below is part of a group roleplay: the user's messages (user) and the characters' replies, labelled by "
        f"speaker:\n{cast}\n"
        "Ignore the speaker_selector lines; they only pick who speaks next. Summarize the scene so far for the "
        "characters who will continue it: who is present, the events in order, relationships, promises and "
        "unresolved threads, and any lasting preferences the user stated. Refer to the characters by name, keep "
        "names and concrete details exact, and write the summary in the conversation's language.\n\n"
        "{conversation_history}"
    )
