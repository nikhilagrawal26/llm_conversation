"""Simulate a conversation between a care coordinator and caregiver."""

from datetime import datetime
from pathlib import Path

from llm_conversation.ai_agent import AIAgent
from pydantic import BaseModel


class MessageFormat(BaseModel):
    """Format for agent responses."""

    content: str


def main() -> None:
    """Run a simulated conversation between care coordinator and caregiver."""
    # Hardcoded prompts
    coordinator_prompt = "You are a care coordinator with experience in special needs (ex. autism)."
    caregiver_prompt = (
        "You are the mom of a 6 year old with Autism. "
        "You are moving to a new state and trying to figure out what districts to look at for your child with special needs. "
        "Your goal is to understand what factors to consider and feel comfortable going into school visits next week."
    )

    # Initialize agents
    print("Initializing agents...")
    coordinator = AIAgent(
        name="Care Coordinator",
        model="gemma3:4b",
        temperature=0.7,
        ctx_size=2048,
        system_prompt=coordinator_prompt,
    )

    caregiver = AIAgent(
        name="Caregiver",
        model="gemma3:4b",
        temperature=0.7,
        ctx_size=2048,
        system_prompt=caregiver_prompt,
    )

    # Conversation storage
    conversation_log: list[tuple[str, str]] = []
    max_turns = 10

    print("\n=== Starting Conversation ===\n")

    # Caregiver starts the conversation
    print(f"[{caregiver.name}]")
    caregiver_first_message = "I am moving to a new state and trying to figure out what districts to look at for my child with special needs. What factors should I consider to get them the best care?"
    print(f"{caregiver_first_message}\n")
    conversation_log.append((caregiver.name, caregiver_first_message))

    # Add caregiver's message to coordinator's history
    coordinator.add_message(name=caregiver.name, role="user", content=caregiver_first_message)

    # Alternate turns
    current_speaker = coordinator
    current_listener = caregiver

    for turn in range(max_turns - 1):  # -1 because caregiver already spoke
        # Generate response
        print(f"[{current_speaker.name}]")
        full_response = ""
        for chunk in current_speaker.get_response(output_format=MessageFormat):
            print(chunk, end="", flush=True)
            full_response += chunk

        print("\n")

        # Log the response
        conversation_log.append((current_speaker.name, full_response))

        # Add response to listener's history
        current_listener.add_message(name=current_speaker.name, role="user", content=full_response)

        # Swap speaker/listener
        current_speaker, current_listener = current_listener, current_speaker

    print("=== Conversation Complete ===\n")

    # Save to markdown
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "traces"
    output_file = output_dir / f"conversation_{timestamp}.md"

    with output_file.open("w") as f:
        f.write(f"# Conversation Log\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model:** gemma3:4b\n")
        f.write(f"**Turns:** {len(conversation_log)}\n\n")
        f.write("---\n\n")

        for speaker, message in conversation_log:
            f.write(f"## {speaker}\n\n")
            f.write(f"{message}\n\n")

    print(f"Conversation saved to: {output_file}")


if __name__ == "__main__":
    main()

