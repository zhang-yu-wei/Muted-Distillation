"""The v1 prompt is frozen; the v2 prompt must describe the v2 corpus."""

import pytest

from officeqa_rl.prompts import CORPUS_PROFILES, build_system_prompt
from officeqa_rl.types import AgentConfig

# sha256 of the system prompt recorded in every OfficeQA result on the shared
# bucket (v1 and v2 alike, since v2 inherited the v1 text). Changing the default
# prompt silently breaks comparability with all of them, so pin it.
V1_PROMPT_SHA256 = "ce3491147fdb531b91e42c2413ef45fe9ee5f307c99951f429c06d6908a6a94d"


def _prompt(profile=None):
    kwargs = {"corpus_profile": profile} if profile else {}
    return build_system_prompt(AgentConfig(step_limit=120, **kwargs))


def test_v1_is_the_default():
    assert AgentConfig(step_limit=120).corpus_profile == "v1"
    assert _prompt() == _prompt("v1")


def test_v1_prompt_is_byte_frozen():
    import hashlib

    digest = hashlib.sha256(_prompt("v1").encode()).hexdigest()
    assert digest == V1_PROMPT_SHA256, (
        "The v1 system prompt changed. Every saved OfficeQA result was produced "
        "with the old text, so this breaks comparability. Update the pin only "
        "together with a deliberate re-evaluation."
    )


def test_v1_prompt_describes_the_v1_corpus():
    prompt = _prompt("v1")
    assert "treasury_bulletin_YYYY_MM.txt" in prompt
    assert "Treasury Bulletin corpus" in prompt


def test_v2_prompt_drops_every_v1_filename_hint():
    """No v1 pattern may survive: none of them match a single v2 file."""
    prompt = _prompt("v2")
    assert "treasury_bulletin" not in prompt
    # The only permitted mention of .txt is the explicit statement that the
    # corpus has none; no .txt glob may be offered as a search pattern.
    assert "officeqa_corpus/treasury" not in prompt
    assert ".txt`" not in prompt.replace("there are no `.txt` files", "")


def test_v2_prompt_names_the_real_v2_families():
    prompt = _prompt("v2")
    assert "combined_statement__historical__cs-YYYY.json" in prompt
    assert "combined_statement__modern__YYYY__<section>.json" in prompt
    assert "govinfo_receipts__YYYY__GOVPUB-<id>.json" in prompt


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown corpus profile"):
        build_system_prompt(AgentConfig(step_limit=120, corpus_profile="v3"))


@pytest.mark.parametrize("profile", sorted(CORPUS_PROFILES))
def test_every_profile_renders_the_shared_scaffolding(profile):
    prompt = _prompt(profile)
    assert "officeqa_corpus/" in prompt
    assert 'submit "<answer>"' in prompt
    assert "{" not in prompt and "}" not in prompt  # no unrendered placeholders


def test_v2txt_matches_the_text_rendering():
    prompt = _prompt("v2txt")
    assert "combined_statement__historical__cs-YYYY.txt" in prompt
    assert ".json" not in prompt and "treasury_bulletin" not in prompt
