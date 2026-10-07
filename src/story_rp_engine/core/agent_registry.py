from typing import Dict, List, Optional, Tuple
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.runners import Runner
from google.adk import Workflow
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model
from story_rp_engine.storage.store import EngineStore
from story_rp_engine.core.types import CharacterCard, GroupCard, Lorebook
from story_rp_engine.rp.agent import create_rp_agent


class AgentRegistry:
    """Central registry ensuring agents, workflows, and runners are created once and reused."""

    def __init__(self, config: EngineConfig, store: EngineStore):
        self.config = config
        self.store = store
        self._rp_agents: Dict[str, LlmAgent] = {}
        self._rp_runners: Dict[str, Runner] = {}
        # Card each store-backed agent was built from. Agents missing here were
        # registered or built from an explicit card and are never refreshed.
        self._rp_agent_cards: Dict[str, CharacterCard] = {}
        self._story_workflow: Optional[Workflow] = None
        self._story_runner: Optional[Runner] = None
        # Group runners with the (group, member cards) they were built from.
        self._group_runners: Dict[str, Tuple[Tuple[GroupCard, List[CharacterCard]], Runner]] = {}

    async def get_or_create_rp_agent(
        self,
        char_id: str,
        card: Optional[CharacterCard] = None,
        lorebook: Optional[Lorebook] = None,
    ) -> LlmAgent:
        cached = self._rp_agents.get(char_id)
        if cached is not None and (card is not None or char_id not in self._rp_agent_cards):
            return cached

        if card is not None:
            agent = create_rp_agent(card, self.config, lorebook=lorebook)
        else:
            # The store may be shared with other instances, so re-read the card and
            # rebuild the agent if the character was edited or deleted elsewhere.
            stored = await self.store.get_character(char_id)
            if not stored:
                self.forget_rp_agent(char_id)
                raise ValueError(f"Character {char_id} not found")
            if cached is not None and self._rp_agent_cards.get(char_id) == stored:
                return cached
            agent = create_rp_agent(stored, self.config, lorebook=lorebook)
            self._rp_agent_cards[char_id] = stored

        self._rp_agents[char_id] = agent
        self._rp_runners.pop(char_id, None)
        return agent

    def register_rp_agent(self, char_id: str, agent: LlmAgent) -> None:
        self._rp_agents[char_id] = agent
        self._rp_agent_cards.pop(char_id, None)
        self._rp_runners.pop(char_id, None)

    def forget_rp_agent(self, char_id: str) -> None:
        """Drops the cached agent and runner for a character (e.g. after deletion)."""
        self._rp_agents.pop(char_id, None)
        self._rp_agent_cards.pop(char_id, None)
        self._rp_runners.pop(char_id, None)

    def _build_compaction_config(self) -> Optional[EventsCompactionConfig]:
        """Constructs ADK EventsCompactionConfig with an explicit summarizer model if enabled."""
        if not self.config.compaction_enabled:
            return None

        model = get_adk_model(self.config)
        kwargs = {}
        if self.config.compaction_prompt_template:
            kwargs["prompt_template"] = self.config.compaction_prompt_template
        summarizer = LlmEventSummarizer(llm=model, **kwargs)

        return EventsCompactionConfig(
            token_threshold=self.config.compaction_token_threshold,
            event_retention_size=self.config.compaction_event_retention_size,
            compaction_interval=self.config.compaction_interval,
            overlap_size=self.config.compaction_overlap_size,
            summarizer=summarizer,
        )

    def _turn_count_compaction_config(self, prompt_template: str) -> Optional[EventsCompactionConfig]:
        """Compaction for workflow apps: turn-count only. Token-threshold compaction also runs before each model
        call, so it could fire between two agents of one turn and summarize away what the first one wrote."""
        if not self.config.compaction_enabled:
            return None
        return EventsCompactionConfig(
            compaction_interval=self.config.compaction_interval,
            overlap_size=self.config.compaction_overlap_size,
            summarizer=LlmEventSummarizer(
                llm=get_adk_model(self.config),
                prompt_template=self.config.compaction_prompt_template or prompt_template,
            ),
        )

    async def get_or_create_rp_runner(self, char_id: str) -> Runner:
        """Retrieves or creates a cached ADK Runner for the specified character agent."""
        # Resolving the agent first evicts the runner if the agent had to be rebuilt.
        agent = await self.get_or_create_rp_agent(char_id)
        if char_id in self._rp_runners:
            return self._rp_runners[char_id]

        compaction_config = self._build_compaction_config()
        app = App(
            name="rp_app",
            root_agent=agent,
            events_compaction_config=compaction_config,
        )
        runner = Runner(
            app=app,
            session_service=self.store.session_service,
            auto_create_session=True,
        )
        self._rp_runners[char_id] = runner
        return runner

    def register_rp_runner(self, char_id: str, runner: Runner) -> None:
        self.register_rp_agent(char_id, runner.agent)
        self._rp_runners[char_id] = runner

    def get_story_workflow(self) -> Workflow:
        if self._story_workflow is None:
            from story_rp_engine.story.workflow import create_story_workflow
            self._story_workflow = create_story_workflow(self.config)
        return self._story_workflow

    def register_story_workflow(self, workflow: Workflow) -> None:
        self._story_workflow = workflow
        self._story_runner = None

    def get_story_runner(self) -> Runner:
        """Retrieves or creates a cached ADK Runner for the story workflow."""
        if self._story_runner is None:
            from story_rp_engine.story.workflow import STORY_SUMMARY_PROMPT
            workflow = self.get_story_workflow()
            compaction_config = self._turn_count_compaction_config(STORY_SUMMARY_PROMPT)
            app = App(
                name="story_app",
                root_agent=workflow,
                events_compaction_config=compaction_config,
            )
            self._story_runner = Runner(
                app=app,
                session_service=self.store.session_service,
                auto_create_session=True,
            )
        return self._story_runner

    def register_story_runner(self, runner: Runner) -> None:
        self._story_runner = runner

    async def get_or_create_group_runner(self, group_id: str) -> Runner:
        """Retrieves or creates the ADK Runner for a group chat.

        The store may be shared with other instances, so the group and its member cards are re-read each time and
        the runner is rebuilt when they changed. Members whose cards were deleted are left out.
        """
        group = await self.store.get_group(group_id)
        if group is None:
            self.forget_group(group_id)
            raise ValueError(f"Group {group_id} not found")
        cards = []
        for char_id in group.char_ids:
            card = await self.store.get_character(char_id)
            if card is not None:
                cards.append(card)
        if not cards:
            raise ValueError(f"Group {group_id} has no characters")

        cached = self._group_runners.get(group_id)
        if cached is not None and cached[0] == (group, cards):
            return cached[1]

        from story_rp_engine.group.prompt_builder import group_summary_prompt
        from story_rp_engine.group.workflow import create_group_workflow
        app = App(
            name="group_app",
            root_agent=create_group_workflow(group, cards, self.config),
            events_compaction_config=self._turn_count_compaction_config(group_summary_prompt(cards)),
        )
        runner = Runner(app=app, session_service=self.store.session_service, auto_create_session=True)
        self._group_runners[group_id] = ((group, cards), runner)
        return runner

    def forget_group(self, group_id: str) -> None:
        """Drops the cached runner for a group (e.g. after deletion)."""
        self._group_runners.pop(group_id, None)
