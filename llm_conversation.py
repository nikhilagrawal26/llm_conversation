from dataclasses import dataclass, field
from typing import cast
import ollama
from rich.text import Text
from rich.console import Console
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter


@dataclass
class AIAgent:
    name: str
    system_prompt: str
    model: str
    temperature: float = 0.8
    ctx_size: int = 2048
    _messages: list[dict[str, str]] = field(init=False)

    def __post_init__(self):
        self._messages = [{"role": "system", "content": self.system_prompt}]

    @property
    def messages(self) -> list[dict[str, str]]:
        return self._messages

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        response = ollama.chat(
            model=self.model,
            messages=self.messages,
            options={
                "num_ctx": self.ctx_size,
                "temperature": self.temperature,
            },
        )

        assistant_reply: str = response["message"]["content"]
        self.messages.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply


@dataclass
class ConversationManager:
    console: Console = field(default_factory=Console)
    agent1: AIAgent | None = None
    agent2: AIAgent | None = None
    color1: str = "blue"
    color2: str = "green"
    initial_message: str = ""

    @staticmethod
    def get_available_models() -> list[str]:
        return [x.model or "" for x in ollama.list().models if x.model]

    def setup_conversation(self) -> None:
        available_models = self.get_available_models()

        self.console.print("\n=== Available Models ===", style="bold cyan")

        for model in available_models:
            self.console.print(Text("• " + model))

        self.console.print("")

        model_completer = WordCompleter(available_models, ignore_case=True)
        model_name = (
            prompt(
                f"Enter model name (default: {available_models[0]}): ",
                completer=model_completer,
                complete_while_typing=True,
            )
            or available_models[0]
        )

        while model_name not in available_models:
            self.console.print("Invalid model name!", style="bold red")
            model_name = (
                prompt(
                    f"Enter model name (default: {available_models[0]}): ",
                    completer=model_completer,
                    complete_while_typing=True,
                )
                or available_models[0]
            )

        while True:
            try:
                temperature_str: str = (
                    prompt("Enter temperature (default: 0.8): ") or "0.8"
                )
                temperature: float = float(temperature_str)
                if not (0.0 <= temperature <= 1.0):
                    raise ValueError("Temperature must be between 0.0 and 1.0")
                break
            except ValueError as e:
                self.console.print(f"Invalid input: {e}", style="bold red")

        while True:
            try:
                ctx_size_str: str = (
                    prompt("Enter context size (default: 2048): ") or "2048"
                )
                ctx_size: int = int(ctx_size_str)
                if ctx_size < 0:
                    raise ValueError("Context size must be a non-negative integer")
                break
            except ValueError as e:
                self.console.print(f"Invalid input: {e}", style="bold red")

        self.console.print("")

        name_1 = prompt("Enter name for AI 1 (default: AI 1): ") or "AI 1"
        name_2 = prompt("Enter name for AI 2 (default: AI 2): ") or "AI 2"
        self.console.print("")
        system_prompt_1 = prompt(f"Enter system prompt for {name_1}: ")
        self.console.print("")
        system_prompt_2 = prompt(f"Enter system prompt for {name_2}: ")
        self.console.print("")
        self.initial_message = prompt("Enter initial message: ", default="Hello")

        self.agent1 = AIAgent(
            name=name_1,
            system_prompt=system_prompt_1,
            model=model_name,
            temperature=temperature,
            ctx_size=ctx_size,
        )

        self.agent2 = AIAgent(
            name=name_2,
            system_prompt=system_prompt_2,
            model=model_name,
            temperature=temperature,
            ctx_size=ctx_size,
        )

    def save_conversation(self, filename: str):
        if not (self.agent1 and self.agent2):
            raise RuntimeError("Conversation not initialized")

        with open(filename, "w", encoding="utf-8") as f:
            _ = f.write(f"=== Details ===\n\n")
            _ = f.write(f"Model: {self.agent1.model}\n")
            _ = f.write(f"Temperature: {self.agent1.temperature}\n")
            _ = f.write(f"Context Size: {self.agent1.ctx_size}\n\n")
            _ = f.write(f"=== Agents ===\n\n")
            _ = f.write(
                f"Name: {self.agent1.name}\nSystem Prompt: {self.agent1.system_prompt}\n\n"
            )
            _ = f.write(
                f"Name: {self.agent2.name}\nSystem Prompt: {self.agent2.system_prompt}\n\n"
            )
            _ = f.write(f"=== Conversation ===\n\n")

            for i, msg in enumerate(self.agent1.messages[1:]):
                # Add separator between messages, except for the first message.
                if i > 0:
                    _ = f.write("\n" + "\u2500" * 80 + "\n\n")

                agent_name = (
                    self.agent1.name if msg["role"] == "assistant" else self.agent2.name
                )
                _ = f.write(f"{agent_name}: {msg['content']}\n")

    def display_message(self, agent_name: str, color: str, message: str):
        text = Text()
        _ = text.append(f"{agent_name}:", style=f"{color} bold")
        _ = text.append(f" {message}")
        self.console.print(text)

    def run_conversation(self):
        if not (self.agent1 and self.agent2):
            raise RuntimeError("Conversation not initialized")

        self.console.print("\n=== Conversation Started ===\n", style="bold cyan")

        self.agent1.messages.append(
            {"role": "assistant", "content": self.initial_message}
        )
        self.display_message(self.agent1.name, self.color1, self.initial_message)
        last_message = self.initial_message
        is_agent1_turn = False

        try:
            while True:
                current_agent = self.agent1 if is_agent1_turn else self.agent2
                last_message = current_agent.chat(last_message)
                self.console.print("")
                self.console.rule()
                self.console.print("")
                self.display_message(
                    current_agent.name,
                    self.color1 if is_agent1_turn else self.color2,
                    last_message,
                )
                is_agent1_turn = not is_agent1_turn
        except KeyboardInterrupt:
            self.console.print("\n=== Conversion Ended ===\n", style="bold cyan")
            self.save_conversation("messages.txt")
            self.console.print(
                "\nConversation saved to messages.txt\n\n", style="bold yellow"
            )
            exit(0)


def main():
    manager = ConversationManager()
    manager.setup_conversation()
    manager.run_conversation()


if __name__ == "__main__":
    main()
