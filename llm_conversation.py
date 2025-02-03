from copy import deepcopy
from dataclasses import dataclass, field
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
        return deepcopy(self._messages)

    def chat(self, user_input: str) -> str:
        self._messages.append({"role": "user", "content": user_input})

        response = ollama.chat(
            model=self.model,
            messages=self._messages,
            options={
                "num_ctx": self.ctx_size,
                "temperature": self.temperature,
            },
        )

        assistant_reply: str = response["message"]["content"]
        self._messages.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply


def get_available_models() -> list[str]:
    return [x.model or "" for x in ollama.list().models if x.model]


def create_ai_agent(console: Console, agent_number: int) -> AIAgent:
    console.print(f"=== Creating AI Agent {agent_number} ===", style="bold cyan")

    available_models = get_available_models()
    console.print("\nAvailable Models:", style="bold")
    for model in available_models:
        console.print(Text("• " + model))
    console.print("")

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
        console.print("Invalid model name!", style="bold red")
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
            temperature_str: str = prompt("Enter temperature (default: 0.8): ") or "0.8"
            temperature: float = float(temperature_str)
            if not (0.0 <= temperature <= 1.0):
                raise ValueError("Temperature must be between 0.0 and 1.0")
            break
        except ValueError as e:
            console.print(f"Invalid input: {e}", style="bold red")

    while True:
        try:
            ctx_size_str: str = prompt("Enter context size (default: 2048): ") or "2048"
            ctx_size: int = int(ctx_size_str)
            if ctx_size < 0:
                raise ValueError("Context size must be a non-negative integer")
            break
        except ValueError as e:
            console.print(f"Invalid input: {e}", style="bold red")

    name = prompt(f"Enter name (default: AI {agent_number}): ") or f"AI {agent_number}"
    system_prompt = prompt(f"Enter system prompt for {name}: ")

    return AIAgent(
        name=name,
        system_prompt=system_prompt,
        model=model_name,
        temperature=temperature,
        ctx_size=ctx_size,
    )


@dataclass
class ConversationManager:
    agent1: AIAgent
    agent2: AIAgent
    initial_message: str
    console: Console = field(default_factory=Console, repr=False)
    color1: str = "blue"
    color2: str = "green"

    def save_conversation(self, filename: str):
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

            for i, msg in enumerate(self.agent1.messages[1:]):
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
        self.console.print("=== Conversation Started ===\n", style="bold cyan")

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


def main():
    console = Console()
    console.clear()
    agent1 = create_ai_agent(console, 1)
    console.clear()
    agent2 = create_ai_agent(console, 2)
    console.clear()
    initial_message = prompt("Enter initial message: ", default="Hello")
    console.clear()

    manager = ConversationManager(
        agent1=agent1, agent2=agent2, initial_message=initial_message
    )
    manager.run_conversation()


if __name__ == "__main__":
    main()
