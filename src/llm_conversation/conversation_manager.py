from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from .ai_agent import AIAgent


@dataclass
class ConversationManager:
    class ConversationLogItem(TypedDict):
        agent: str
        content: str

    # TODO: Extend this to support more than two agents.
    agent1: AIAgent
    agent2: AIAgent
    initial_message: str | None
    use_markdown: bool = False
    allow_termination: bool = False
    _conversation_log: list[ConversationLogItem] = field(
        default_factory=list, init=False
    )

    def __post_init__(self):
        # Modify system prompt to include termination instructions if allowed
        instruction: str = ""

        if self.use_markdown:
            instruction += (
                "\n\nYou may use Markdown for text formatting. "
                "Examples: *italic*, **bold**, `code`, [link](https://example.com), etc."
            )

        # TODO: Make the <TERMINATE> token prompt stronger. Right now the agent often confuses it or forgets about it
        # and can't use it properly.
        if self.allow_termination:
            instruction += (
                "\n\nYou may terminate the conversation with the `<TERMINATE>` token "
                "if you believe it has reached a natural conclusion. "
                "Do not include the token in your message otherwise."
            )

        self.agent1.system_prompt += instruction
        self.agent2.system_prompt += instruction

    # TODO: Support JSON output.
    def save_conversation(self, filename: Path):
        with open(filename, "w", encoding="utf-8") as f:
            _ = f.write(f"=== Agent 1 ===\n\n")
            _ = f.write(f"Name: {self.agent1.name}\n")
            _ = f.write(f"Model: {self.agent1.model}\n")
            _ = f.write(f"Temperature: {self.agent1.temperature}\n")
            _ = f.write(f"Context Size: {self.agent1.ctx_size}\n")
            _ = f.write(f"System Prompt: {self.agent1.system_prompt}\n\n")
            _ = f.write(f"=== Agent 2 ===\n\n")
            _ = f.write(f"Name: {self.agent2.name}\n")
            _ = f.write(f"Model: {self.agent2.model}\n")
            _ = f.write(f"Temperature: {self.agent2.temperature}\n")
            _ = f.write(f"Context Size: {self.agent2.ctx_size}\n")
            _ = f.write(f"System Prompt: {self.agent2.system_prompt}\n\n")
            _ = f.write(f"=== Conversation ===\n\n")

            for i, msg in enumerate(self._conversation_log):
                if i > 0:
                    _ = f.write("\n" + "\u2500" * 80 + "\n\n")

                _ = f.write(f"{msg['agent']}: {msg['content']}\n")

    def run_conversation(self) -> Iterator[tuple[str, Iterator[str]]]:
        """
        Generate an iterator of conversation responses.

        Yields:
            Iterator of (agent_name, message) tuples or None
        """

        last_message = self.initial_message
        is_agent1_turn = True

        # If a non-empty initial message is provided, start with it.
        if self.initial_message is not None:
            # Make the first agent the one to say the initial message, and the second agent the one to respond.
            self.agent1.add_message("assistant", self.initial_message)
            self._conversation_log.append(
                {"agent": self.agent1.name, "content": self.initial_message}
            )
            yield (self.agent1.name, iter([self.initial_message]))
            is_agent1_turn = False

        while True:
            current_agent = self.agent1 if is_agent1_turn else self.agent2
            response_stream = current_agent.chat(last_message)
            last_message_chunks: list[str] = []

            def stream_chunks() -> Iterator[str]:
                for chunk in response_stream:
                    last_message_chunks.append(chunk)
                    yield chunk

            yield (current_agent.name, stream_chunks())

            # stream_chunks must be exhausted before this function is called again.
            assert next(response_stream, None) is None

            last_message = "".join(last_message_chunks).strip()
            self._conversation_log.append(
                {"agent": current_agent.name, "content": last_message}
            )

            # Check for termination token.
            # TODO: Filter out the `<TERMINATE>` token from the output to prevent it from displaying.
            if self.allow_termination and "<TERMINATE>" in last_message:
                break

            is_agent1_turn = not is_agent1_turn
