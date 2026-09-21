from google.adk import Workflow
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import pytest
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.story.director_agent import create_director_agent
from story_rp_engine.story.writer_agent import create_writer_agent
from story_rp_engine.story.workflow import create_story_workflow


def test_create_director_and_writer_agents():
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    director = create_director_agent(config)
    writer = create_writer_agent(config)

    assert director.name == "story_director"
    assert "{premise?}" in director.instruction
    assert "{genre?}" in director.instruction
    assert "{tone?}" in director.instruction
    assert "{current_text?}" in director.instruction

    assert writer.name == "story_writer"
    assert "{premise?}" in writer.instruction
    assert "{genre?}" in writer.instruction
    assert "{tone?}" in writer.instruction
    assert "{current_text?}" in writer.instruction


def test_create_story_workflow_graph_structure():
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    wf = create_story_workflow(config)

    assert isinstance(wf, Workflow)
    assert wf.name == "story_workflow"
    assert len(wf.graph.nodes) == 3  # __START__, story_director, story_writer
    node_names = [n.name for n in wf.graph.nodes]
    assert "story_director" in node_names
    assert "story_writer" in node_names

    # Verify graph edge sequence: START -> director -> writer
    edge_pairs = [(e.from_node.name, e.to_node.name) for e in wf.graph.edges]
    assert ("__START__", "story_director") in edge_pairs
    assert ("story_director", "story_writer") in edge_pairs


def test_format_story_input_removed_from_workflow():
    import story_rp_engine.story.workflow as wf_mod
    assert not hasattr(wf_mod, "format_story_input")
    assert not hasattr(wf_mod, "format_story_director_input")


@pytest.mark.anyio
async def test_session_state_injection_into_workflow_instructions():
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    wf = create_story_workflow(config)

    captured_instructions = {}

    class MockCaptureLlm(BaseLlm):
        model: str = "mock"
        agent_name: str = ""

        async def generate_content_async(self, llm_request, stream=False):
            captured_instructions[self.agent_name] = llm_request.config.system_instruction
            yield LlmResponse(
                partial=False,
                content=types.Content(parts=[types.Part.from_text(text=f"{self.agent_name} output")]),
            )

    for node in wf.graph.nodes:
        if node.name in ["story_director", "story_writer"]:
            node.model = MockCaptureLlm(agent_name=node.name)

    app = App(name="story_app", root_agent=wf)
    runner = Runner(app=app, session_service=InMemorySessionService(), auto_create_session=True)

    state_delta = {
        "premise": "A dragon sleeps in the cave.",
        "genre": "Fantasy",
        "tone": "Epic",
        "current_text": "The torch flickers in the damp air.",
        "instruction": "Wake the dragon.",
    }
    content = types.Content(role="user", parts=[types.Part.from_text(text="Wake the dragon.")])
    async for _ in runner.run_async(
        user_id="User",
        session_id="test_story_state_injection",
        new_message=content,
        state_delta=state_delta,
    ):
        pass

    director_inst = captured_instructions["story_director"]
    assert "Premise: A dragon sleeps in the cave." in director_inst
    assert "Genre: Fantasy" in director_inst
    assert "Tone: Epic" in director_inst
    assert "The torch flickers in the damp air." in director_inst

    writer_inst = captured_instructions["story_writer"]
    assert "A dragon sleeps in the cave." in writer_inst
    assert "Fantasy" in writer_inst
    assert "Epic" in writer_inst
    assert "The torch flickers in the damp air." in writer_inst


