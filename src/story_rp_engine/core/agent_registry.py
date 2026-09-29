from typing import Dict, Optional
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.runners import Runner
from google.adk import Workflow
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.model_provider import get_adk_model
from story_rp_engine.storage.store import EngineStore
from story_rp_engine.core.types import CharacterCard, Lorebook
from story_rp_engine.rp.agent import create_rp_agent


class AgentRegistry:
    """Central registry ensuring agents, workflows, and runners are created once and reused."""

    def __init__(self, config: EngineConfig, store: EngineStore):
        self.config = config
        self.store = store
        self._rp_agents: Dict[str, LlmAgent] = {}
        self._rp_runners: Dict[str, Runner] = {}
        self._story_workflow: Optional[Workflow] = None
        self._story_runner: Optional[Runner] = None

    def get_or_create_rp_agent(
        self,
        char_id: str,
        card: Optional[CharacterCard] = None,
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

    def get_or_create_rp_runner(self, char_id: str) -> Runner:
        """Retrieves or creates a cached ADK Runner for the specified character agent."""
        if char_id in self._rp_runners:
            return self._rp_runners[char_id]

        agent = self.get_or_create_rp_agent(char_id)
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
            workflow = self.get_story_workflow()
            compaction_config = self._build_compaction_config()
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
