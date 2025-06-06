from pydantic import BaseModel, Discriminator
from typing import Any, Literal, Annotated, Union, Optional, Callable
from functools import wraps
from .._base import State

class TextContent(BaseModel):
    type: Literal["text"]
    value: str

class MediaContent(BaseModel):
    type: Literal["image", "audio", "video"]
    # index of the media in the respective list (mostly for content-reusability)
    index: int
    # e.g. <|image|>, <|image_{index}|>, etc.
    # Note that this is very model-specific, so using this for a model
    # it wasn't designed for may lead to unexpected results.
    # TODO: provide some kind of interface for doing "conversions"?
    text_representation: str

Content = Annotated[
    Union[TextContent, MediaContent],
    Discriminator("type")
]

class EngineMessage(BaseModel):
    role: str
    content: list[Content]

class EngineState(State):
    def __init__(self) -> None:
        super().__init__()
        self.active_message: Optional[EngineMessage] = None
        self.messages: list[EngineMessage] = []
        self.images: list[Any] = []
        self.audio: list[Any] = []
        self.videos: list[Any] = []

    @property
    def active_role(self) -> Optional[str]:
        return self.active_message.role if self.active_message else None

    @active_role.setter
    def active_role(self, role: Optional[str]) -> None:
        if role is None:
            if self.active_message is not None:
                self.messages.append(self.active_message)
                self.active_message = None
            return

        if self.active_message is None:
            self.active_message = EngineMessage(role=role, content=[])
        else:
            raise ValueError("Cannot set active role while another message is active.")

    def add_text(self, text: str) -> None:
        """
        Adds text to the current active message.
        :param text: The text to add.
        :raises ValueError: If there is no active message.
        """
        if self.active_message is None:
            raise ValueError("Cannot add text without an active message.")
        if len(self.active_message.content) > 0 and self.active_message.content[-1].type == "text":
            # If the last content is text, append to it
            self.active_message.content[-1].value += text
        else:
            # Otherwise, create a new text content
            self.active_message.content.append(TextContent(type="text", value=text))

    def add_media(self, media_type: Literal["image", "audio", "video"], media: Any, text_representation: Union[str, Callable[[int], str]], allow_ref: bool) -> None:
        """
        Adds media to the current active message.
        :param media_type: Type of the media, one of "image", "audio", or "video".
        :param media: The media object to add (e.g., PIL Image, audio file, video file).
        :param text_representation: A string or a callable that generates the text representation for the media.
            If a callable is provided, it should accept an index and return a string representation.
            For example, a callable could be `lambda ix: f"<|image_{ix}|>"`.
        :param allow_ref: If True, allows reusing existing media by reference; otherwise, adds a new entry.
        :raises ValueError: If there is no active message or if the media type is unsupported.
        """
        if self.active_message is None:
            raise ValueError("Cannot add image without an active message.")

        if media_type == "image":
            arr = self.images
        elif media_type == "audio":
            arr = self.audio
        elif media_type == "video":
            arr = self.videos
        else:
            raise ValueError(f"Unsupported media type: {media_type}")

        if allow_ref and media in arr:
            index = arr.index(media)
        else:
            index = len(arr)
            arr.append(media)

        self.active_message.content.append(
            MediaContent(
                type=media_type,
                index=index,
                text_representation=text_representation(index) if callable(text_representation) else text_representation,
            )
        )
