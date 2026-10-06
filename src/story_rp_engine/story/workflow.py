from google.adk import Context, Workflow
from google.adk.workflow import node
from story_rp_engine.core.config import EngineConfig
from story_rp_engine.story.director_agent import create_director_agent
from story_rp_engine.story.writer_agent import create_writer_agent

STORY_SUMMARY_PROMPT = (
    "Below is part of a story-writing session: the user's instructions (user), the Director's notes "
    "(story_director) and the passages the Writer produced (story_writer). Summarize the story so far for a "
    "writer who will continue it: characters (names, appearances, relationships), places, the events in order, "
    "promises and unresolved threads, the point of view, tense and style, and any lasting preferences the user "
    "stated. Keep names and concrete details exact. Leave out the Director's planning unless it became part of "
    "the story. Write the summary in the story's language.\n\n"
    "{conversation_history}"
)


def create_story_workflow(config: EngineConfig) -> Workflow:
    """Creates the story workflow: the director reads the whole (compacted) conversation and writes notes
    for the writer, who sees only those notes and the story state."""
    director = create_director_agent(config)
    writer = create_writer_agent(config)

    @node(rerun_on_resume=True)
    async def story_turn(ctx: Context):
        # No input: the director reads the user's message from the conversation history.
        notes = await ctx.run_node(director)
        return await ctx.run_node(writer, notes)

    return Workflow(name="story_workflow", edges=[("START", story_turn)])
