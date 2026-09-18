from google.adk import Workflow
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import StoryRequest
from story_rp_engine.story.director_agent import create_director_agent
from story_rp_engine.story.writer_agent import create_writer_agent
from story_rp_engine.story.workflow import (
    create_story_workflow,
    format_story_input,
)


def test_create_director_and_writer_agents():
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    director = create_director_agent(config)
    writer = create_writer_agent(config)

    assert director.name == "story_director"
    assert "framing" in director.instruction.lower()
    assert writer.name == "story_writer"
    assert "prose" in writer.instruction.lower()


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


def test_format_story_input():
    req = StoryRequest(
        premise="A dragon sleeps in the cave.",
        current_text="The torch flickers.",
        instruction="Wake the dragon.",
        genre="Fantasy",
        tone="Epic",
    )
    prompt = format_story_input(req)
    assert "Premise: A dragon sleeps in the cave." in prompt
    assert "Genre: Fantasy" in prompt
    assert "Tone: Epic" in prompt
    assert "Current Text:\nThe torch flickers." in prompt
    assert "User Instruction: Wake the dragon." in prompt
    assert "Provide brief scene framing and narrative guidance for the writer." in prompt


def test_format_story_input_defaults():
    req = StoryRequest(
        current_text="Just some text",
    )
    prompt = format_story_input(req)
    assert "Premise: Not specified" in prompt
    assert "User Instruction: Continue the story naturally from the current point." in prompt

    req_none_instruction = StoryRequest(
        current_text="Just some text",
        instruction=None,
    )
    prompt_none = format_story_input(req_none_instruction)
    assert "User Instruction: Continue the story." in prompt_none


