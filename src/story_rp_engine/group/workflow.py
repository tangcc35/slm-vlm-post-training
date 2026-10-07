from typing import Dict, List, Optional
from google.adk import Context, Workflow
from google.adk.workflow import node
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import CharacterCard, GroupCard
from story_rp_engine.group.char_agent import create_group_char_agent
from story_rp_engine.group.prompt_builder import group_agent_name
from story_rp_engine.group.selector_agent import create_speaker_selector

# Characters run on this branch and the selector on a sub-branch of it. ADK shows an agent the events of its own
# branch and its ancestors, so the selector sees the characters' lines but the characters never see its JSON.
# A node with no branch would see everything, so the characters need the named branch too.
SCENE_BRANCH = "group"


def pick_speakers(plan: Optional[dict], member_ids: List[str]) -> List[str]:
    """The selector's speakers without repeats, or every member in group order if it picked no one."""
    speakers = list(dict.fromkeys((plan or {}).get("speakers", [])))
    return speakers or list(member_ids)


def group_speakers(group: GroupCard) -> Dict[str, str]:
    """Maps each member's agent name (the author of its events) to its char_id."""
    return {group_agent_name(char_id): char_id for char_id in group.char_ids}


def create_group_workflow(group: GroupCard, cards: List[CharacterCard], config: EngineConfig) -> Workflow:
    """One group chat turn: the speaker selector picks who replies, then each chosen character replies in order,
    seeing the user's message and the replies before it."""
    selector = create_speaker_selector(group, cards, config)
    agents = {c.char_id: create_group_char_agent(c, group, cards, config) for c in cards}

    @node(rerun_on_resume=True)
    async def group_turn(ctx: Context):
        plan = await ctx.run_node(selector, use_sub_branch=True, override_branch=SCENE_BRANCH)
        for char_id in pick_speakers(plan, list(agents)):
            await ctx.run_node(agents[char_id], override_branch=SCENE_BRANCH)

    return Workflow(name="group_workflow", edges=[("START", group_turn)])
