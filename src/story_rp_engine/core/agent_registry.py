from typing import Dict, Optional
from google.adk.agents import LlmAgent
from google.adk import Workflow
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.storage.store import EngineStore
from story_rp_engine.core.types import CharacterCardV2, Lorebook
from story_rp_engine.rp.agent import create_rp_agent


class AgentRegistry:
    """Central registry ensuring agents and workflows are created once and reused."""

    def __init__(self, config: EngineConfig, store: EngineStore):
        self.config = config
        self.store = store
        self._rp_agents: Dict[str, LlmAgent] = {}
        self._story_workflow: Optional[Workflow] = None

    def get_or_create_rp_agent(
        self,
        char_id: str,
        card: Optional[CharacterCardV2] = None,
        lorebook: Optional[Lorebook] = None,
    ) -> LlmAgent:
        if char_id in self._rp_agents:
            return self._rp_agents[char_id]

        if card is None:
            card = self.store.get_character(char_id)
        if not card:
            raise ValueError(f"Character {char_id} not found")

        agent = create_rp_agent(card, self.config, lorebook=lorebook)
        self._rp_agents[char_id] = agent
        return agent

    def register_rp_agent(self, char_id: str, agent: LlmAgent) -> None:
        self._rp_agents[char_id] = agent

    def get_story_workflow(self) -> Workflow:
        if self._story_workflow is None:
            from story_rp_engine.story.workflow import create_story_workflow
            self._story_workflow = create_story_workflow(self.config)
        return self._story_workflow

    def register_story_workflow(self, workflow: Workflow) -> None:
        self._story_workflow = workflow
