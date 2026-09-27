"""Validation tests for repository agent skills framework (.agents/skills)."""

import os
import re
import pytest
import yaml

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "..", ".agents", "skills")
EXPECTED_SKILLS = [
    "unsloth-sft",
    "grpo-reasoning-rl",
    "model-export-gguf",
    "character-rp",
    "story-copilot",
    "story-rp-backend",
    "story-rp-frontend",
]


def test_skills_directory_exists():
    assert os.path.isdir(SKILLS_DIR), f"Skills directory not found at {SKILLS_DIR}"


@pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
def test_skill_structure_and_frontmatter(skill_name):
    skill_path = os.path.join(SKILLS_DIR, skill_name)
    assert os.path.isdir(skill_path), f"Skill directory missing: {skill_name}"

    skill_md_path = os.path.join(skill_path, "SKILL.md")
    assert os.path.isfile(skill_md_path), f"SKILL.md missing in {skill_name}"

    with open(skill_md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify YAML frontmatter delimiters
    assert content.startswith("---\n"), f"SKILL.md in {skill_name} must start with YAML frontmatter"
    parts = content.split("---\n", 2)
    assert len(parts) >= 3, f"SKILL.md in {skill_name} must have closing frontmatter delimiter"

    frontmatter = yaml.safe_load(parts[1])
    assert isinstance(frontmatter, dict), f"Frontmatter in {skill_name} is not a valid dict"

    # Frontmatter name assertions
    assert "name" in frontmatter, f"'name' missing in {skill_name} frontmatter"
    assert frontmatter["name"] == skill_name, f"Frontmatter name '{frontmatter['name']}' != directory '{skill_name}'"
    assert re.match(r"^[a-z0-9-]+$", frontmatter["name"]), f"Invalid skill name format: {frontmatter['name']}"
    assert len(frontmatter["name"]) <= 64, f"Skill name exceeds 64 chars in {skill_name}"

    # Frontmatter description assertions
    assert "description" in frontmatter, f"'description' missing in {skill_name} frontmatter"
    desc = frontmatter["description"]
    assert desc.startswith("Use when"), f"Description in {skill_name} must start with 'Use when...'"
    assert len(desc) <= 1024, f"Description in {skill_name} exceeds 1024 chars"


@pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
def test_openai_agent_yaml(skill_name):
    openai_yaml_path = os.path.join(SKILLS_DIR, skill_name, "agents", "openai.yaml")
    assert os.path.isfile(openai_yaml_path), f"agents/openai.yaml missing in {skill_name}"

    with open(openai_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "interface" in data, f"'interface' missing in {openai_yaml_path}"
    interface = data["interface"]
    assert "display_name" in interface and interface["display_name"]
    assert "short_description" in interface and interface["short_description"]
    assert "default_prompt" in interface and interface["default_prompt"]


@pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
def test_reference_links_resolve(skill_name):
    skill_path = os.path.join(SKILLS_DIR, skill_name)
    skill_md_path = os.path.join(skill_path, "SKILL.md")

    if not os.path.isfile(skill_md_path):
        pytest.skip(f"SKILL.md does not exist yet for {skill_name}")

    with open(skill_md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all reference links pointing to references/ (supports backticks, markdown links, etc.)
    ref_matches = re.findall(r"references/([a-zA-Z0-9_-]+\.md)", content)
    assert len(ref_matches) > 0, f"Skill {skill_name} must reference at least one guide in references/"

    for ref_file in ref_matches:
        ref_path = os.path.join(skill_path, "references", ref_file)
        assert os.path.isfile(ref_path), f"Referenced file missing: {ref_path} in skill {skill_name}"
