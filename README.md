# AI Math Assistant

A small AI agent that solves arithmetic questions written in plain English, such as *"add 25 and 15, then multiply by 2"*. It uses **LangChain tool calling** with an **OpenAI** model. The model doesn't do the arithmetic itself: it breaks the problem into steps and calls Python tools for each one.

```
You: add 25 and 15, then multiply by 2, then divide by 4
Assistant: 20
```

## Features

- Four math tools: `add`, `subtract`, `multiply` and `divide`. Dividing by zero returns an error message instead of crashing.
- Handles multi-step problems by calling the tools one step at a time.
- Interactive chat in the terminal.
- Local tracing with `--trace` shows each tool call and its result.
- LangSmith tracing (optional) gives you a web dashboard of prompts, tool calls, tokens and timing.
- Stops after 20 agent steps, so it can't loop forever.

## Project structure

```
.
├── math_assistant.py                            # The assistant (run this)
├── AI_Math_Assistant_Instructions.pdf           # Step-by-step guide with the full code
├── AI-Math-Assistant_Tool_Calling_OpenAI.ipynb  # Original lab notebook (for learning)
├── pyproject.toml / uv.lock                     # Dependencies (managed by uv)
└── .env                                         # Your API keys (not committed)
```

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

1. Install the dependencies:

   ```powershell
   uv sync
   ```

2. Create a `.env` file in the project folder:

   ```
   OPENAI_API_KEY=sk-...your key...
   ```

## Usage

```powershell
uv run python math_assistant.py
```

Type a question and press Enter. To exit, type `quit`, `exit` or `q`, or press **Ctrl+C**. Ctrl+C also stops it in the middle of an answer.

Example questions:

- `what is 100 minus 37?`
- `multiply 12.5 by 4`
- `add 25 and 15, then multiply by 2, then divide by 4`
- `divide 10 by 0`

## Tracing

### Local trace

Add `--trace` to print every tool call the agent makes:

```powershell
uv run python math_assistant.py --trace
```

```
You: add 25 and 15, then multiply by 2
  [trace] call   add({'a': 25, 'b': 15})
  [trace] result add -> 40.0
  [trace] call   multiply({'a': 40, 'b': 2})
  [trace] result multiply -> 80.0
Assistant: The result is 80.
```

### LangSmith

1. Create a free account at [smith.langchain.com](https://smith.langchain.com) and create an API key.
2. Add these lines to `.env`:

   ```
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=lsv2_...your key...
   LANGSMITH_PROJECT=ai-math-assistant
   ```

3. Run the assistant as usual. On startup it prints `LangSmith trace: on`, and each question appears as a trace in the `ai-math-assistant` project.

To turn LangSmith off, set `LANGSMITH_TRACING=false`. Traces are uploaded to LangSmith, so don't type sensitive information while it's on.

## How it works

| Part | Role |
|------|------|
| `@tool` functions | Turn Python functions into tools. The type hints and docstring tell the model what each tool does and which arguments it takes. |
| `ChatOpenAI` | The language model that reads the question and decides which tool to call. |
| `create_agent` | Connects the model and the tools and runs the call-tool, read-result loop. |
| `SYSTEM_PROMPT` | Tells the model to always use the tools for arithmetic and to answer in plain text. |
| `ask()` | Streams one question through the agent, prints trace lines if enabled, and returns the final answer. |

## Customization

**Change the model:** edit `OPENAI_MODEL` at the top of `math_assistant.py`. Any OpenAI chat model that supports tool calling works, for example `gpt-4o-mini`.

**Add a tool:** write a function, decorate it with `@tool`, give it a docstring, and add it to `TOOLS`:

```python
@tool
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return base ** exponent

TOOLS = [add, subtract, multiply, divide, power]
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `OPENAI_API_KEY not found` | Create `.env` in the project folder and run the command from that folder. |
| `ModuleNotFoundError` | Run with `uv run python ...` so the project's `.venv` is used, or run `uv sync`. |
| `401` / model not found | Check your API key and the `OPENAI_MODEL` name. |
| No traces in LangSmith | Check that `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set in `.env`. |

## Security

`.env` is listed in `.gitignore`, so your API keys are never committed. Don't put keys directly in the code or the notebook.
