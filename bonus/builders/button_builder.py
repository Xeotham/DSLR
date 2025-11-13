
from customtkinter import *
from typing import NewType, Callable, Literal


class ButtonBuilder:
    def __init__(self, app):
        self.main_app = app
        self.current_button = CTkButton(self.main_app)

    def new(self):
        self.current_button = CTkButton(self.main_app)
        return self

    def size(self, width: int, height: int):
        self.current_button.configure(
            width=width,
            height=height
        )
        return self

    def corner_radius(self, radius: int):
        self.current_button.configure(
            corner_radius=radius
        )
        return self

    def border(self, width: int, spacing: int = 2, color: tuple[str] | None = None):
        self.current_button.configure(
            border_width=width,
            border_spacing=spacing,
        )
        if color:
            self.current_button.configure(border_color=color)
        return self

    def text(self, content: str, color: tuple[str] | None = None):
        self.current_button.configure(
            text=content,
        )
        if color:
            self.current_button.configure(text_color=color)
        return self

    def hover(self, toggled: bool, color: tuple[str] | None = None):
        self.current_button.configure(
            hover=toggled,
        )
        if color:
            self.current_button.configure(hover_color=color)
        return self

    def fg_color(self, color: str):
        self.current_button.configure(
            fg_color=color
        )
        return self

    def command(self, command: Callable):
        self.current_button.configure(
            command=command,
        )
        return self

    def grid(self, row: int, column: int, padx: int=20, pady: int=20):
        self.current_button.grid(
            row=row,
            column=column,
            padx=padx,
            pady=pady,
        )
        return self

    def pack(
        self,
        side: Literal["top", "bottom", "left", "right"] = "top",
        padx: int = 0,
        pady: int = 0,
        fill: Literal["x", "y", "both", "none"] = "none",
        expand: bool = False,
        anchor: Literal["center", "n", "s", "e", "w"] = "center",
    ):
        if fill == "none":
            fill = None
        self.current_button.pack(
            side=side,
            padx=padx,
            pady=pady,
            fill=fill,
            expand=expand,
            anchor=anchor
        )
        return self

    def build(self):
        return self.current_button
