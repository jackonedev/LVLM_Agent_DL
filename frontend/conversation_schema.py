# pylint: disable=missing-module-docstring
import base64
import dataclasses
import os
from enum import Enum, auto
from io import BytesIO
from typing import Any, Dict, List


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history"""

    system: str
    roles: List[str]
    messages: List[List[str]]
    map_roles: Dict[str, str]
    version: str = "Unknown"
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "\n"

    def _get_prompt_role(self, role):
        if self.map_roles is not None and role in self.map_roles.keys():
            return self.map_roles[role]
        return role

    def _build_content_for_first_message_in_conversation(
        self, first_message: List[str]
    ):
        content = []
        if len(first_message) != 2:
            raise TypeError(
                "First message in Conversation needs \
                    to include a prompt and a base64-enconded image!"
            )

        prompt, b64_image = first_message[0], first_message[1]

        # handling prompt
        if prompt is None:
            raise TypeError("API does not support None prompt yet")
        content.append({"type": "text", "text": prompt})
        if b64_image is None:
            raise TypeError("API does not support text only conversation yet")

        # handling image
        if not self.isBase64(b64_image):
            raise TypeError(
                "Image in Conversation's first message must be stored under base64 encoding!"
            )

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": b64_image,
                },
            }
        )
        return content

    def _build_content_for_follow_up_messages_in_conversation(
        self, follow_up_message: List[str]
    ):

        if follow_up_message is not None and len(follow_up_message) > 1:
            raise TypeError(
                "Follow-up message in Conversation must not include an image!"
            )

        # handling text prompt
        if follow_up_message is None or follow_up_message[0] is None:
            raise TypeError(
                "Follow-up message in Conversation must include exactly one text message"
            )

        text = follow_up_message[0]
        return text
        
    def _isBase64(self, sb):
        try:
            if isinstance(sb, str):
                # 1st Edge Case: If there's any unicode here,
                # an exception will be thrown and the function will return false
                sb_bytes = bytes(sb, "ascii")
            elif isinstance(sb, bytes):
                sb_bytes = sb
            else:
                raise ValueError("Argument must be string or bytes")
            return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Error: {e} in function: utils.isBase64", end="\n")
            return False

    def get_message(self):
        messages = self.messages
        api_messages = []
        for i, msg in enumerate(messages):
            role, message_content = msg
            if i == 0:
                # get content for very first message in conversation
                content = self._build_content_for_first_message_in_conversation(
                    message_content
                )
            else:
                # get content for follow-up message in conversation
                content = self._build_content_for_follow_up_messages_in_conversation(
                    message_content
                )

            api_messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )
        return api_messages

    # this method helps represent a multi-turn chat into as a single turn chat format
    def serialize_messages(self):
        messages = self.messages
        ret = ""
        if self.sep_style == SeparatorStyle.SINGLE:
            if self.system is not None and self.system != "":
                ret = self.system + self.sep
            for i, (role, message) in enumerate(messages):
                role = self._get_prompt_role(role)
                if message:
                    if isinstance(message, List):
                        # get prompt only
                        message = message[0]
                    if i == 0:
                        # do not include role at the beginning
                        ret += message
                    else:
                        ret += role + ": " + message
                    if i < len(messages) - 1:
                        # avoid including sep at the end of serialized message
                        ret += self.sep
                else:
                    ret += role + ":"
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        if len(self.messages) == 0:
            # data verification for the very first message
            assert (
                role == self.roles[0]
            ), f"the very first message in conversation must be from role {self.roles[0]}"
            assert (
                len(message) == 2
            ), "the very first message in conversation must include both prompt and an image"
            prompt, image = message[0], message[1]
            assert prompt is not None, "prompt must be not None"
            assert self.isBase64(image), "image must be under base64 encoding"
        else:
            # data verification for follow-up message
            assert (
                role in self.roles
            ), f"the follow-up message must be from one of the roles {self.roles}"
            assert (
                len(message) == 1
            ), "the follow-up message must consist of one text message only, no image"

        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            version=self.version,
            map_roles=self.map_roles,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": [[x, y[0] if len(y) == 1 else y] for x, y in self.messages],
            "version": self.version,
        }



