import pytest
from unittest.mock import MagicMock
from google.genai import types
from google.adk.models import LlmRequest
from google.adk.agents.callback_context import CallbackContext
from story_rp_engine.core.types import Lorebook, LorebookEntry
from story_rp_engine.rp.callbacks import create_rp_before_model_callback


def test_before_model_callback_injects_lore_and_authors_note():
    lore = Lorebook(
        name="Fantasy World",
        entries=[
            LorebookEntry(keys=["sword"], content="Excalibur is a legendary blade."),
        ],
    )
    cb = create_rp_before_model_callback(lorebook=lore)

    mock_context = MagicMock(spec=CallbackContext)
    mock_context.state = {"authors_note": "Tone: mysterious"}

    request = LlmRequest(
        model="test-model",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="I draw my silver sword!")]
            )
        ],
        config=types.GenerateContentConfig(system_instruction="Base character prompt.")
    )

    res = cb(mock_context, request)
    assert res is None  # Allow generation to proceed

    # Verify lore and author's note were injected into system_instruction or dynamic instruction
    instruction = str(request.config.system_instruction)
    assert "Excalibur is a legendary blade." in instruction
    assert "Tone: mysterious" in instruction


def test_before_model_callback_no_lorebook_and_no_state():
    cb = create_rp_before_model_callback(lorebook=None)
    mock_context = MagicMock(spec=CallbackContext)
    mock_context.state = {}

    request = LlmRequest(
        model="test-model",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="Hello there")]
            )
        ],
        config=types.GenerateContentConfig(system_instruction="Base prompt")
    )

    res = cb(mock_context, request)
    assert res is None
    assert request.config.system_instruction == "Base prompt"


def test_before_model_callback_lore_only_without_authors_note():
    lore = Lorebook(
        name="SciFi World",
        entries=[
            LorebookEntry(keys=["hyperdrive"], content="Hyperdrive allows FTL travel."),
        ],
    )
    cb = create_rp_before_model_callback(lorebook=lore)
    mock_context = MagicMock(spec=CallbackContext)
    mock_context.state = {}

    request = LlmRequest(
        model="test-model",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="Engage the hyperdrive!")]
            )
        ],
        config=types.GenerateContentConfig(system_instruction="Base prompt")
    )

    res = cb(mock_context, request)
    assert res is None
    instruction = str(request.config.system_instruction)
    assert "Hyperdrive allows FTL travel." in instruction
    assert "### Narrative Directive" not in instruction


def test_before_model_callback_authors_note_only_without_lore():
    cb = create_rp_before_model_callback(lorebook=None)
    mock_context = MagicMock(spec=CallbackContext)
    mock_context.state = {"authors_note": "Keep replies under 2 sentences."}

    request = LlmRequest(
        model="test-model",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="Hello")]
            )
        ],
        config=types.GenerateContentConfig(system_instruction="Base prompt")
    )

    res = cb(mock_context, request)
    assert res is None
    instruction = str(request.config.system_instruction)
    assert "Keep replies under 2 sentences." in instruction
    assert "### Relevant World Information" not in instruction


def test_before_model_callback_default_config_no_system_instruction():
    cb = create_rp_before_model_callback(lorebook=None)
    mock_context = MagicMock(spec=CallbackContext)
    mock_context.state = {"authors_note": "Tone: dark"}

    request = LlmRequest(
        model="test-model",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="Hello")]
            )
        ]
    )

    res = cb(mock_context, request)
    assert res is None
    assert request.config is not None
    assert "Tone: dark" in str(request.config.system_instruction)
