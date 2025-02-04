from copy import deepcopy

import ollama


class AIAgent:
    name: str
    model: str
    temperature: float = 0.8
    ctx_size: int = 2048
    _messages: list[dict[str, str]]

    def __init__(
        self,
        name: str,
        model: str,
        temperature: float,
        ctx_size: int,
        system_prompt: str,
    ):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.ctx_size = ctx_size
        self._messages = [{"role": "system", "content": system_prompt}]

    @property
    def messages(self) -> list[dict[str, str]]:
        return deepcopy(self._messages)

    @property
    def system_prompt(self) -> str:
        return self._messages[0]["content"]

    @system_prompt.setter
    def system_prompt(self, value: str):
        self._messages[0]["content"] = value

    def add_message(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})

    def chat(self, user_input: str | None) -> str:
        # `None` user_input means the agent is starting the conversation or responding multiple times.
        if user_input is not None:
            self.add_message("user", user_input)

        # TODO: Stream the conversation instead of sending all of the messages at once.
        response = ollama.chat(
            model=self.model,
            messages=self._messages,
            options={
                "num_ctx": self.ctx_size,
                "temperature": self.temperature,
            },
        )

        assistant_reply: str = response["message"]["content"]
        self.add_message("assistant", assistant_reply)
        return assistant_reply
