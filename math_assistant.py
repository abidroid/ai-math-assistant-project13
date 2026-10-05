"""AI Math Assistant: a LangChain agent with add, subtract, multiply and divide tools.

Run from the project folder:
    uv run python math_assistant.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

OPENAI_MODEL = "gpt-5.4-nano"  # any tool-calling OpenAI chat model works, e.g. "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Math tools
# ---------------------------------------------------------------------------

@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return a + b."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first and return a - b."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return a * b."""
    return a * b


@tool
def divide(a: float, b: float) -> float | str:
    """Divide the first number by the second and return a / b.
    Returns an error message if b is zero."""
    if b == 0:
        return "Error: cannot divide by zero."
    return a / b


TOOLS = [add, subtract, multiply, divide]

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Always use the provided tools for arithmetic "
    "instead of calculating in your head. For multi-step problems, call the tools one "
    "step at a time, then give the final answer in one short plain-text sentence (no LaTeX)."
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def build_agent():
    load_dotenv()  # reads OPENAI_API_KEY from the .env file
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Add it to the .env file in the project folder.")

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    return create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


def ask(agent, question: str) -> str:
    # recursion_limit stops the agent if it ever gets stuck in a loop of tool calls
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 20},
    )
    return result["messages"][-1].content


def main():
    agent = build_agent()
    print("AI Math Assistant (type 'quit' to exit)")
    print("Example: add 25 and 15, then multiply by 2\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        try:
            print("Assistant:", ask(agent, question), "\n")
        except Exception as e:
            print("Error:", e, "\n")

    print("Goodbye!")


if __name__ == "__main__":
    main()
