# pylint: disable=missing-module-docstring
import base64
import dataclasses
import os
from enum import Enum, auto
from io import BytesIO
from typing import Any, List

from utils import PROMPT_TEMPLATE, encode_image, prediction_guard_llava_conv


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()


# pylint: disable=too-many-instance-attributes
@dataclasses.dataclass
class GradioInstance:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "\n"
    sep2: str = None
    version: str = "Unknown"
    path_to_img: str = None
    video_title: str = None
    path_to_video: str = None
    caption: str = None
    mm_rag_chain: Any = None

    skip_next: bool = False

    def _template_caption(self):
        out = ""
        if self.caption is not None:
            out = f"The caption associated with the image is '{self.caption}'. "
        return out

    # pylint: disable=missing-function-docstring
    def get_prompt_for_rag(self):
        messages = self.messages
        assert len(messages) == 2, "length of current conversation should be 2"
        assert (
            messages[1][1] is None
        ), "the first response message of current conversation should be None"
        ret = messages[0][1]
        return ret

    # pylint: disable=missing-function-docstring
    def get_conversation_for_lvlm(self):

        pg_conv = prediction_guard_llava_conv.copy()
        image_path = self.path_to_img
        b64_img = encode_image(image_path)
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if msg is None:
                break
            if i == 0:
                pg_conv.append_message(
                    prediction_guard_llava_conv.roles[0], [msg, b64_img]
                )
            elif i == len(self.messages[self.offset :]) - 2:
                pg_conv.append_message(
                    role,
                    [PROMPT_TEMPLATE.format(transcript=self.caption, user_query=msg)],
                )
            else:
                pg_conv.append_message(role, [msg])
        return pg_conv

    def append_message(self, role, message):
        self.messages.append([role, message])

    # pylint: disable=unused-argument

    def get_images(self, return_pil=False):
        images = []
        if self.path_to_img is not None:
            path_to_image = self.path_to_img
            images.append(path_to_image)
        return images

    # pylint: disable=too-many-locals
    # pylint: disable=fixme
    # TODO: Muchas variables locales
    # TODO: Refactorizar este método
    def to_gradio_chatbot(self):
        ret = []
        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if isinstance(msg, tuple):
                    msg, image, _ = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    width, height = image.size
                    if height > width:
                        height, width = longest_edge, shortest_edge
                    else:
                        height, width = shortest_edge, longest_edge
                    image = image.resize((width, height))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace("<image>", "").strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return GradioInstance(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
            mm_rag_chain=self.mm_rag_chain,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "path_to_img": self.path_to_img,
            "video_title": self.video_title,
            "path_to_video": self.path_to_video,
            "caption": self.caption,
        }

    def get_path_to_subvideos(self):
        if self.video_title is not None and self.path_to_img is not None:
            # pylint: disable=undefined-variable
            info = video_helper_map[self.video_title]
            path = info["path"]
            prefix = info["prefix"]
            vid_index = self.path_to_img.split("/")[-1]
            vid_index = vid_index.split("_")[-1]
            vid_index = vid_index.replace(".jpg", "")
            ret = f"{prefix}{vid_index}.mp4"
            ret = os.path.join(path, ret)
            return ret
        if self.path_to_video is not None:
            return self.path_to_video
        return None
