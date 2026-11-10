"""Prompts shared by OfficeQA inference and training agents."""

from officeqa_rl.types import AgentConfig


SYSTEM_PROMPT = """You are an autonomous OfficeQA agent and an expert in answering questions related to the U.S. Treasury and economy.

The {corpus_name} corpus is available in the workspace at `officeqa_corpus/`. Use filesystem search to locate relevant documents and sections; these documents are long, so do not read the whole corpus or whole files unless needed.

If multiple documents report the same metric, use the most up-to-date revision unless the question specifies an exact date or document. Do not rely on the first matching value you encounter.

You can issue exactly one command per turn. Put the command in exactly one fenced {action_block_language} block and do not write anything after the block.

Available actions:
- Run shell commands such as `ls`, `find`, `grep`/`rg`, `sed`, `head`, `cat`, and `python3` to search, read evidence, and calculate.
- `submit "<answer>"`: Submit the final answer and finish the task.

Useful corpus patterns:
{corpus_patterns}
- Use Python for arithmetic, statistics, regressions, and careful unit conversion.

REQUIRED ANSWER FORMAT:
When you have the final answer, call `submit "<answer>"` with only the answer value inside the quotes.

Example:
```{action_block_language}
{grep_example}
```

Example:
```{action_block_language}
submit "123.45"
```

Use maximum precision in calculations unless the question specifies otherwise. Never ask follow-up questions."""


# Corpus-specific prompt fragments. OfficeQA v1 ships the Treasury Bulletin
# corpus as `treasury_bulletin_YYYY_MM.txt`; v2 ships a different corpus of
# 1,435 Combined Statement and govinfo receipt JSONs, where none of the v1
# filename hints match anything. Keying the fragments off the corpus keeps each
# evaluation's hints true of the files the agent can actually see, and leaves
# the v1 prompt byte-identical to every result already on record.
CORPUS_PROFILES = {
    "v1": {
        "corpus_name": "Treasury Bulletin",
        "corpus_patterns": (
            "- Filenames look like `treasury_bulletin_YYYY_MM.txt`.\n"
            "- Prefer targeted searches such as `grep -Rni -A 3 -B 3 \"phrase\" "
            "officeqa_corpus/treasury_bulletin_1954_*.txt`."
        ),
        "grep_example": (
            'grep -Rni -A 2 -B 2 "national defense" '
            "officeqa_corpus/treasury_bulletin_1941_01.txt | head -50"
        ),
    },
    "v2": {
        "corpus_name": "U.S. Treasury Combined Statement and receipts",
        "corpus_patterns": (
            "- Every document is a `.json` file; there are no `.txt` files.\n"
            "- Filenames look like `combined_statement__historical__cs-YYYY.json` "
            "(1872-1994), `combined_statement__modern__YYYY__<section>.json` "
            "(2001-2024), and `govinfo_receipts__YYYY__GOVPUB-<id>.json` (1793-1893).\n"
            "- The `<section>` and `<id>` parts are not guessable: match on the year "
            "and let the glob cover the rest.\n"
            "- Prefer targeted searches such as `grep -Rni -A 3 -B 3 \"phrase\" "
            "officeqa_corpus/combined_statement__*1954*.json`."
        ),
        "grep_example": (
            'grep -Rni -A 2 -B 2 "national defense" '
            "officeqa_corpus/combined_statement__*1941*.json | head -50"
        ),
    },
    # The same v2 corpus rendered to plain text by prepare_officeqa_pro_v2.py
    # (transformed/*.txt, identical stems). Kept separate so the hints match the
    # extension the agent will actually see.
    "v2txt": {
        "corpus_name": "U.S. Treasury Combined Statement and receipts",
        "corpus_patterns": (
            "- Every document is a plain-text `.txt` file.\n"
            "- Filenames look like `combined_statement__historical__cs-YYYY.txt` "
            "(1872-1994), `combined_statement__modern__YYYY__<section>.txt` "
            "(2001-2024), and `govinfo_receipts__YYYY__GOVPUB-<id>.txt` (1793-1893).\n"
            "- The `<section>` and `<id>` parts are not guessable: match on the year "
            "and let the glob cover the rest.\n"
            "- Prefer targeted searches such as `grep -Rni -A 3 -B 3 \"phrase\" "
            "officeqa_corpus/combined_statement__*1954*.txt`."
        ),
        "grep_example": (
            'grep -Rni -A 2 -B 2 "national defense" '
            "officeqa_corpus/combined_statement__*1941*.txt | head -50"
        ),
    },
}


WEB_SEARCH_NOTE = (
    "\n\nYou also have access to web search if your sandbox runtime or generation "
    "backend provides it; use it only for external values needed by the question."
)


def build_system_prompt(config: AgentConfig) -> str:
    """Render the system prompt from explicit agent configuration."""
    try:
        profile = CORPUS_PROFILES[config.corpus_profile]
    except KeyError:
        raise ValueError(
            f"Unknown corpus profile {config.corpus_profile!r}; "
            f"expected one of {sorted(CORPUS_PROFILES)}"
        ) from None
    prompt = SYSTEM_PROMPT.format(
        action_block_language=config.action_block_language,
        **profile,
    )
    if config.web_search_instruction:
        prompt += WEB_SEARCH_NOTE
    return prompt


def format_error_observation(config: AgentConfig, error: Exception) -> str:
    """Tell the model how to repair an invalid action response."""
    return (
        "Format error: respond with exactly one fenced command block, "
        f"for example:\n```{config.action_block_language}\nsubmit \"123.45\"\n```"
    )


__all__ = [
    "CORPUS_PROFILES",
    "SYSTEM_PROMPT",
    "WEB_SEARCH_NOTE",
    "build_system_prompt",
    "format_error_observation",
]
