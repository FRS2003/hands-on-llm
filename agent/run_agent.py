"""
Entry point for the Academic Research Agent.

Usage:
    # On the server (with GPU, FAISS, BGE-M3):
    python run_agent.py

    # Streamlit UI:
    streamlit run app.py

    # CLI test with a single question:
    python run_agent.py --question "Compare the data augmentation methods in VSD classification papers"
"""

import argparse
import os

from openai import OpenAI

from agent_core import ResearchAgent, ToolRegistry
from tools import register_all_tools


def create_agent() -> ResearchAgent:
    """Create a fully configured ResearchAgent with all tools registered."""
    api_key = os.environ.get("DEEPSEEK_API_KEY","")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    registry = ToolRegistry()
    register_all_tools(registry)

    return ResearchAgent(tools=registry, llm_client=client)


def run_cli(question: str):
    """Run the agent from CLI and print the reasoning trace."""
    print(f"\n{'='*70}")
    print(f"Research Question: {question}")
    print(f"{'='*70}\n")

    agent = create_agent()

    def print_step(step):
        print(f"┌─ Step {agent.state.step_count if hasattr(agent, 'state') else '?'}")
        print(f"│  Thought: {step.thought[:100]}...")
        if step.action:
            print(f"│  Action:  {step.action}")
            print(f"│  Input:   {step.action_input[:80] if step.action_input else 'N/A'}...")
            obs = (step.observation or "")[:200]
            print(f"│  Result:  {obs}...")
        else:
            print(f"│  FINAL ANSWER:")
            for line in (step.observation or "").split("\n")[:5]:
                print(f"│  {line[:100]}")
        print(f"└─")

    state = agent.run(question, stream_callback=print_step)

    print(f"\n{'='*70}")
    print(f"Agent completed in {len(state.steps)} steps.")
    print(f"Tools called: {[s.action for s in state.steps if s.action]}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Academic Research Agent")
    parser.add_argument("--question", "-q", type=str,
                        default="What are the main data augmentation techniques used in VSD ultrasound classification?",
                        help="Research question to answer")
    args = parser.parse_args()

    run_cli(args.question)
