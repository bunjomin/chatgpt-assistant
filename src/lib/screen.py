import os
import asyncio
import logging

from rich.console import Console
from rich.control import Control
from rich.markdown import Markdown


class Screen:
    def __init__(self):
        self.console = Console(width=os.get_terminal_size().columns)
        self.control = Control()
        self.console.clear()
        self.console.show_cursor(False)

    def render(self, text):
        self.console.clear()
        self.console.print(Markdown(text, justify="left"), justify="left")

    async def append(self, message, messages):
        control_sequences = ["#", "**", "*", "[](", "__", "_"]
        width, height = os.get_terminal_size()
        x = len(messages[-1])
        self.console.print(
            self.control.move_to(
                x,
                (height // 2) - max(len(messages), len("".join(messages).split("\n"))),
            ),
            justify="left",
        )
        for j, char in enumerate(message):
            sequence = next(
                (seq for seq in control_sequences if char == seq),
                None,
            )
            if sequence:
                continue
            await asyncio.sleep(0.01)
            self.console.print(char, justify="left")

    async def write(self, messages):
        try:
            control_sequences = ["#", "**", "*", "[](", "__", "_"]
            width, height = os.get_terminal_size()
            total = ""
            self.console.clear()
            for i, message in enumerate(messages):
                for j, char in enumerate(message):
                    sequence = next(
                        (
                            seq
                            for seq in control_sequences
                            if message.startswith(seq, j)
                        ),
                        None,
                    )
                    if sequence:
                        total += char
                        continue
                    total += char
                    self.console.print(
                        self.control.move_to(0, i == 0 and 0 or height - height + i),
                        justify="left",
                    )
                    self.console.line(height // 2 - len(total.split("\n")))
                    await asyncio.sleep(0.01)
                    self.console.print(Markdown(total, justify="left"), justify="left")
                total += "\n"

            self.console.clear()
            self.console.print(
                self.control.move_to(0, i == 0 and 0 or height - height + i),
                justify="left",
            )
            self.console.line(height // 2 - len(total.split("\n")))
            self.console.print(Markdown(total, justify="left"), justify="left")

        except Exception as e:
            logging.error(f"print: Exception: {e}")

    def quit(self):
        self.console.clear()
        self.console.show_cursor(True)
