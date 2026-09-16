from unittest.mock import MagicMock, patch
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.core.types import StoryRequest
from story_rp_engine.story.director_agent import create_director_agent
from story_rp_engine.story.writer_agent import create_writer_agent
from story_rp_engine.story.workflow import expand_story, stream_expand_story


def test_create_director_and_writer_agents():
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    director = create_director_agent(config)
    writer = create_writer_agent(config)

    assert director.name == "story-director"
    assert "framing" in director.instruction.lower()
    assert writer.name == "story-writer"
    assert "prose" in writer.instruction.lower()


def test_expand_story_pipeline():
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    request = StoryRequest(
        premise="A detective arrives at a quiet harbor.",
        current_text="Fog covered the docks.",
        instruction="Describe his arrival and first impression.",
        genre="Noir Mystery",
        tone="Dark and brooding",
    )

    with patch(
        "story_rp_engine.story.director_agent.LlmAgent.invoke",
        return_value="Focus on cold rain and solitary footsteps.",
    ):
        with patch(
            "story_rp_engine.story.writer_agent.LlmAgent.invoke",
            return_value="He stepped into the mist, collar turned against the damp chill.",
        ):
            prose = expand_story(request, config)
            assert "mist" in prose


def test_expand_story_prompt_assembly_and_framing_flow():
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    request = StoryRequest(
        premise="Spaceship approaching an unknown derelict.",
        current_text="Sensors pinged with erratic signals.",
        instruction="Detail docking procedure.",
        genre="Sci-Fi Thriller",
        tone="Tense and claustrophobic",
    )

    mock_director_invoke = MagicMock(return_value="Emphasize silent mechanical groans and fluctuating air pressure.")
    mock_writer_invoke = MagicMock(return_value="The airlock clamped with a heavy shudder, echoing into silence.")

    with patch("story_rp_engine.story.director_agent.LlmAgent.invoke", mock_director_invoke):
        with patch("story_rp_engine.story.writer_agent.LlmAgent.invoke", mock_writer_invoke):
            prose = expand_story(request, config)
            assert prose == "The airlock clamped with a heavy shudder, echoing into silence."

            # Verify director received the story details
            director_prompt = mock_director_invoke.call_args[0][0]
            assert "Premise: Spaceship approaching an unknown derelict." in director_prompt
            assert "Genre: Sci-Fi Thriller" in director_prompt
            assert "Tone: Tense and claustrophobic" in director_prompt
            assert "Sensors pinged with erratic signals." in director_prompt
            assert "Detail docking procedure." in director_prompt

            # Verify writer received director's framing
            writer_prompt = mock_writer_invoke.call_args[0][0]
            assert "Director's Guidance: Emphasize silent mechanical groans and fluctuating air pressure." in writer_prompt
            assert "Genre: Sci-Fi Thriller" in writer_prompt
            assert "Tone: Tense and claustrophobic" in writer_prompt
            assert "Detail docking procedure." in writer_prompt


def test_expand_story_with_response_objects_and_defaults():
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    request = StoryRequest(
        premise=None,
        current_text="The tavern fell silent.",
    )

    director_resp = MagicMock()
    director_resp.text = "   Spotlight the hooded stranger in the corner.   "

    writer_resp = MagicMock()
    writer_resp.text = "   A lone figure raised an iron tankard, eyes gleaming in the firelight.   "

    with patch("story_rp_engine.story.director_agent.LlmAgent.invoke", return_value=director_resp):
        with patch("story_rp_engine.story.writer_agent.LlmAgent.invoke", return_value=writer_resp):
            prose = expand_story(request, config)
            assert prose == "A lone figure raised an iron tankard, eyes gleaming in the firelight."


def test_stream_expand_story_pipeline():
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    request = StoryRequest(
        premise="A detective arrives at a quiet harbor.",
        current_text="Fog covered the docks.",
        instruction="Describe his arrival and first impression.",
        genre="Noir Mystery",
        tone="Dark and brooding",
    )

    with patch(
        "story_rp_engine.story.director_agent.LlmAgent.invoke",
        return_value="Focus on cold rain and solitary footsteps.",
    ):
        with patch(
            "story_rp_engine.story.writer_agent.LlmAgent.invoke",
            return_value="He stepped into the mist.",
        ):
            chunks = list(stream_expand_story(request, config))
            assert len(chunks) > 1
            assert "".join(chunks) == "He stepped into the mist."


def test_stream_expand_story_none_response():
    config = EngineConfig(model_name="ollama/llama3.1:8b")
    request = StoryRequest(
        premise="Test",
        current_text="...",
    )

    with patch(
        "story_rp_engine.story.director_agent.LlmAgent.invoke",
        return_value="framing",
    ):
        with patch(
            "story_rp_engine.story.writer_agent.LlmAgent.invoke",
            return_value=None,
        ):
            chunks = list(stream_expand_story(request, config))
            assert chunks == []

