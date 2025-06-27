from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, Optional, Union

from pydantic import BaseModel, Discriminator
from typing_extensions import assert_never

from ..._schema import TokenUsage
from .._base import State

if TYPE_CHECKING:
    from ._engine import Engine
    from ._tokenizer import ChatMessage


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


Content = Annotated[Union[TextContent, MediaContent], Discriminator("type")]


class EngineMessage(BaseModel):
    role: str
    content: list[Content]


def stringify_content(content: list[Content]) -> str:
    s = ""
    for item in content:
        if isinstance(item, TextContent):
            s += item.value
        elif isinstance(item, MediaContent):
            s += item.text_representation
        else:
            if TYPE_CHECKING:
                assert_never(item)
            raise TypeError(f"Unsupported content type: {type(item)}")
    return s


class EngineState(ABC, State):
    def __init__(self) -> None:
        # Initialize with zero token usage rather than default None
        # since engine can fast-forward tokens
        super().__init__(token_usage=TokenUsage(ff_tokens=0))
        self.images: list[Any] = []
        self.audio: list[Any] = []
        self.videos: list[Any] = []

    @abstractmethod
    def get_prompt(self, engine: "Engine") -> str:
        """
        Returns the current prompt as a string.
        :param engine: The engine instance to which this state belongs.
        """
        pass

    @abstractmethod
    def _add_content(
        self,
        content: Content,
    ) -> None:
        """Low-level method to add content to the current state."""
        pass

    @abstractmethod
    def open_role(self, role: str) -> None:
        """Opens a new role in the current state.
        :param role: The role to open.
        :raises ValueError: If there is already an active role.
        :raises TypeError: If the model does not support roles.
        """
        pass

    @abstractmethod
    def close_role(self) -> None:
        """Closes the active role in the current state.
        :raises ValueError: If there is no active role to close.
        :raises TypeError: If the model does not support roles.
        """
        pass

    def add_text(self, text: str) -> None:
        """Adds text to the current state.
        :param text: The text to add."""
        if text:
            self._add_content(TextContent(type="text", value=text))

    def add_media(
        self,
        media_type: Literal["image", "audio", "video"],
        media: Any,
        text_representation: Union[str, Callable[[int], str]],
        allow_ref: bool,
    ) -> None:
        """
        Adds media to the current state.
        :param media_type: Type of the media, one of "image", "audio", or "video".
        :param media: The media object to add (e.g., PIL Image, audio file, video file).
        :param text_representation: A string or a callable that generates the text representation for the media.
            If a callable is provided, it should accept an index and return a string representation.
            For example, a callable could be `lambda ix: f"<|image_{ix}|>"`.
        :param allow_ref: If True, allows reusing existing media by reference; otherwise, adds a new entry.
        :raises ValueError: If the media type is unsupported.
        """
        if media_type == "image":
            arr = self.images
        elif media_type == "audio":
            arr = self.audio
        elif media_type == "video":
            arr = self.videos
        else:
            raise ValueError(
                f"Unsupported media type: {media_type}. Supported types are 'image', 'audio', and 'video'."
            )

        if allow_ref and media in arr:
            index = arr.index(media)
        else:
            index = len(arr)
            arr.append(media)

        self._add_content(
            MediaContent(
                type=media_type,
                index=index,
                text_representation=text_representation(index)
                if callable(text_representation)
                else text_representation,
            )
        )


class EngineCompletionState(EngineState):
    def __init__(self) -> None:
        super().__init__()
        self.content: list[Content] = []

    def get_prompt(self, engine: "Engine") -> str:
        return stringify_content(self.content)

    def open_role(self, role: str) -> None:
        # TODO: custom error type and instructions on how to rectify?
        raise TypeError("Cannot open role with a completion model.")

    def close_role(self) -> None:
        # TODO: custom error type and instructions on how to rectify?
        raise TypeError("Cannot close role with a completion model.")

    def _add_content(
        self,
        content: Content,
    ) -> None:
        self.content.append(content)


class EngineChatState(State):
    def __init__(self) -> None:
        super().__init__()
        self.active_message: Optional[EngineMessage] = None
        self.messages: list[EngineMessage] = []

    def get_prompt(self, engine: "Engine") -> str:
        engine_messages = self.messages.copy()
        if self.active_message is not None:
            engine_messages.append(self.active_message)
        chat_messages: list["ChatMessage"] = []
        for message in engine_messages:
            chat_messages.append(
                {
                    "role": message.role,
                    "content": stringify_content(message.content),
                }
            )
        return engine.apply_chat_template(
            messages=chat_messages,
            continue_final_message=(self.active_message is not None),
        )

    def open_role(self, role: str) -> None:
        if self.active_message is not None:
            raise ValueError(
                f"Cannot open role {role} while another role is active: {self.active_message.role}. Close the current role first."
            )
        self.active_message = EngineMessage(role=role, content=[])

    def close_role(self) -> None:
        if self.active_message is None:
            raise ValueError("Cannot close role without an active message.")
        if not self.active_message.content:
            raise ValueError(
                f"Cannot close role {self.active_message.role} with no content. Add content before closing the role."
            )
        self.messages.append(self.active_message)
        self.active_message = None

    def _add_content(
        self,
        content: Content,
    ) -> None:
        if self.active_message is None:
            raise ValueError("Cannot add content without an active message.")
        self.active_message.content.append(content)
