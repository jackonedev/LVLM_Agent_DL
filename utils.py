# pylint: disable=missing-module-docstring
import base64
import dataclasses
import glob
import json
import os
import random
import textwrap
import urllib.parse
from enum import Enum, auto
from io import BytesIO, StringIO
from typing import Any, Dict, Iterator, List, Optional, Sequence, TextIO, Union

# pylint: disable=import-error
import cv2
import PIL
import requests
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import MessageLikeRepresentation
from langchain_core.prompt_values import PromptValue
from PIL import Image
from predictionguard import PredictionGuard
from pytubefix import Stream, YouTube
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import WebVTTFormatter

SERVER_ERROR_MSG = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)

PROMPT_TEMPLATE = (
    """The transcript associated with the image is '{transcript}'. {user_query}"""
)

LANCEDB_HOST_FILE = ...
TBL_NAME = ...


MultimodalModelInput = Union[
    PromptValue, str, Sequence[MessageLikeRepresentation], Dict[str, Any]
]


def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    return get_from_env(key, env_key, default=default)


# pylint: disable=unused-argument
def get_from_env(key: str, env_key: str, default: Optional[str] = None) -> str:
    """Get a value from a dictionary or an environment variable."""
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    return default


# pylint: disable=missing-function-docstring
def load_env():
    _ = load_dotenv(find_dotenv())


# pylint: disable=missing-function-docstring
def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key


# pylint: disable=missing-function-docstring
def get_prediction_guard_api_key():
    load_env()
    prediction_guard_api_key = os.getenv("PREDICTION_GUARD_API_KEY", None)
    if prediction_guard_api_key is None:
        prediction_guard_api_key = input("Please enter your Prediction Guard API Key: ")
    return prediction_guard_api_key


# "https://proxy-dl-itdc.predictionguard.com"
prediction_guard_url_endpoint = os.getenv(
    "DLAI_PREDICTION_GUARD_URL_ENDPOINT", "https://dl-itdc.predictionguard.com"
)


# function helps to prepare list image-text pairs
# from the first [test_size] data of a Huggingface dataset
# Utilizada en L2-*.ipynb para visualizar embeddings en 2D
def prepare_dataset_for_umap_visualization(
    hf_dataset: str, class_name: str, templates: list = None, test_size: int = 1000
) -> list:
    """
    Prepares a dataset for UMAP visualization by generating image-text pairs.

    Args:
        hf_dataset (str): The name of the Huggingface dataset to load.
        class_name (str): The class name to be used in the text templates.
        templates (list, optional): A list of text templates to describe the images.
                                    Defaults to [
                                        "a picture of {}",
                                        "an image of {}",
                                        "a nice {}",
                                        "a beautiful {}"
                                        ].
        test_size (int, optional): The size of the test split. Defaults to 1000.

    Returns:
        list: A list of dictionaries, each containing a 'caption' and a 'pil_img' key.
    """
    if templates is None:
        templates = [
            "a picture of {}",
            "an image of {}",
            "a nice {}",
            "a beautiful {}",
        ]
    # load Huggingface dataset (download if needed)
    dataset = load_dataset(hf_dataset, trust_remote_code=True)
    # split dataset with specific test_size
    train_test_dataset = dataset["train"].train_test_split(test_size=test_size)
    # get the test dataset
    test_dataset = train_test_dataset["test"]
    img_txt_pairs = []
    # pylint: disable=consider-using-enumerate
    for i in range(len(test_dataset)):
        img_txt_pairs.append(
            {
                "caption": templates[random.randint(0, len(templates) - 1)].format(
                    class_name
                ),
                "pil_img": test_dataset[i]["image"],
            }
        )
    return img_txt_pairs


def download_video(video_url, path="/tmp/"):
    print(f"Getting video information for {video_url}")
    if not video_url.startswith("http"):
        return os.path.join(path, video_url)

    filepath = glob.glob(os.path.join(path, "*.mp4"))
    if len(filepath) > 0:
        return filepath[0]

    # pylint: disable=possibly-used-before-assignment
    def progress_callback(
        stream: Stream, data_chunk: bytes, bytes_remaining: int
    ) -> None:
        pbar.update(len(data_chunk))

    yt = YouTube(video_url, on_progress_callback=progress_callback)
    stream = (
        yt.streams.filter(progressive=True, file_extension="mp4", res="720p")
        .desc()
        .first()
    )
    if stream is None:
        stream = (
            yt.streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .desc()
            .first()
        )
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, stream.default_filename)
    if not os.path.exists(filepath):
        print("Downloading video from YouTube...")
        pbar = tqdm(
            desc="Downloading video from YouTube", total=stream.filesize, unit="bytes"
        )
        stream.download(path)
        pbar.close()
    return filepath


def get_video_id_from_url(video_url):
    """
    Examples:
    - http://youtu.be/SA2iWivDJiE
    - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    - http://www.youtube.com/embed/SA2iWivDJiE
    - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """

    url = urllib.parse.urlparse(video_url)
    if url.hostname == "youtu.be":
        return url.path[1:]
    if url.hostname in ("www.youtube.com", "youtube.com"):
        if url.path == "/watch":
            p = urllib.parse.parse_qs(url.query)
            return p["v"][0]
        if url.path[:7] == "/embed/":
            return url.path.split("/")[2]
        if url.path[:3] == "/v/":
            return url.path.split("/")[2]

    return video_url


# if this has transcript then download


def get_transcript_vtt(video_url, path="/tmp"):
    video_id = get_video_id_from_url(video_url)
    filepath = os.path.join(path, "captions.vtt")
    if os.path.exists(filepath):
        return filepath

    transcript = YouTubeTranscriptApi.get_transcript(
        video_id, languages=["en-GB", "en"]
    )
    formatter = WebVTTFormatter()
    webvtt_formatted = formatter.format_transcript(transcript)

    with open(filepath, "w", encoding="utf-8") as webvtt_file:
        webvtt_file.write(webvtt_formatted)
    webvtt_file.close()

    return filepath


# helper function for convert time in second to time format for .vtt or .srt file
def format_timestamp(
    seconds: float, always_include_hours: bool = False, fractional_seperator: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{fractional_seperator}{milliseconds:03d}"


# a help function that helps to convert a specific time written
# as a string in format `webvtt` into a time in miliseconds
def str2time(strtime):
    """
    Convert a time string in the format "HH:MM:SS" to milliseconds.

    Args:
        strtime (str): A string representing time in the format "HH:MM:SS".
                       The string may optionally be enclosed in double quotes.

    Returns:
        float: The corresponding time in milliseconds.

    Example:
        >>> str2time('01:30:45')
        5445000.0
        >>> str2time('"02:15:30"')
        8130000.0
    """
    # strip character " if exists
    strtime = strtime.strip('"')
    # get hour, minute, second from time string
    hrs, mins, seconds = [float(c) for c in strtime.split(":")]
    # get the corresponding time as total seconds
    total_seconds = hrs * 60**2 + mins * 60 + seconds
    total_miliseconds = total_seconds * 1000
    return total_miliseconds


def _process_text(text: str, max_line_width=None):
    if max_line_width is None or max_line_width < 0:
        return text

    lines = textwrap.wrap(text, width=max_line_width, tabsize=4)
    return "\n".join(lines)


# Resizes a image and maintains aspect ratio


def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)


# helper function to convert transcripts generated by whisper to .vtt file


def write_vtt(transcript: Iterator[dict], file: TextIO, max_line_width=None):
    print("WEBVTT\n", file=file)
    for segment in transcript:
        text = _process_text(segment["text"], max_line_width).replace("-->", "->")

        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )


# helper function to convert transcripts generated by whisper to .srt file
def write_srt(transcript: Iterator[dict], file: TextIO, max_line_width=None):
    """
    Write a transcript to a file in SRT format.
    Example usage:
        from pathlib import Path
        from whisper.utils import write_srt
        result = transcribe(model, audio_path, temperature=temperature, **args)
        # save SRT
        audio_basename = Path(audio_path).stem
        with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
    """
    for i, segment in enumerate(transcript, start=1):
        text = _process_text(segment["text"].strip(), max_line_width).replace(
            "-->", "->"
        )

        # write srt lines
        # pylint: disable=fixme
        # TODO: Revisar el funcionamiento de este print
        # pylint: disable=line-too-long
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, fractional_seperator=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, fractional_seperator=',')}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )


def get_subs(
    segments: Iterator[dict], subtitle_format: str, max_line_width: int = -1
) -> str:
    """
    Generate subtitle text from segments in the specified format.

    Args:
        segments (Iterator[dict]):
            An iterator of segment dictionaries containing subtitle data.
        subtitle_format (str):
            The format of the subtitles to generate. Supported formats are "vtt" and "srt".
        max_line_width (int, optional):
            The maximum width of subtitle lines. Defaults to -1, which means no limit.

    Returns:
        str: The generated subtitle text in the specified format.

    Raises:
        Exception: If the specified format is not supported.
    """
    segment_stream = StringIO()
    # pylint: disable=broad-exception-raised
    if subtitle_format not in ["vtt", "srt"]:
        raise Exception("Unknown format " + subtitle_format)

    if subtitle_format == "vtt":
        write_vtt(segments, file=segment_stream, max_line_width=max_line_width)
    elif subtitle_format == "srt":
        write_srt(segments, file=segment_stream, max_line_width=max_line_width)

    segment_stream.seek(0)
    return segment_stream.read()


# encoding image at given path or PIL Image using base64


# pylint: disable=invalid-name
def encode_image(image_path_or_PIL_img: Union[str, PIL.Image.Image]) -> str:
    """
    Encodes an image to a base64 string.

    Args:
        image_path_or_PIL_img (Union[str, PIL.Image.Image]):
            The image to encode. This can be either a file path to the image or a PIL Image object.

    Returns:
        str: The base64 encoded string of the image.
    """
    if isinstance(image_path_or_PIL_img, PIL.Image.Image):
        buffered = BytesIO()
        image_path_or_PIL_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    with open(image_path_or_PIL_img, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# checking whether the given string is base64 or not


def isBase64(sb):
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


def encode_image_from_path_or_url(image_path_or_url):
    try:
        # try to open the url to check valid url
        # pylint: disable=fixme
        # TODO: esto parece una librería que no está importada
        # f = urlopen(image_path_or_url)
        r = base64.b64encode(requests.get(image_path_or_url, timeout=9).content).decode(
            "utf-8"
        )
        return r
    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(f"Error: {e} in function: utils.encode_image_from_path_or_url", end="\n")
        # this is a path to image
        with open(image_path_or_url, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


# helper function to compute the joint embedding
# of a prompt and a base64-encoded image through PredictionGuard
def bt_embedding_from_prediction_guard(prompt: str, base64_image: str) -> list:
    """
    Generates an embedding from a given prompt and an optional base64-encoded image
    using the PredictionGuard client.

    Args:
        prompt (str):
            The text prompt to generate the embedding from.
        base64_image (str):
            The base64-encoded image to include in the embedding generation.
            If None or an empty string, only the prompt is used.

    Returns:
        list: The embedding generated by the PredictionGuard client.

    Raises:
        TypeError: If the provided base64_image is not in valid base64 encoding.
    """
    # get PredictionGuard client
    client = _getPredictionGuardClient()
    message = {
        "text": prompt,
    }
    if base64_image is not None and base64_image != "":
        if not isBase64(base64_image):
            raise TypeError("image input must be in base64 encoding!")
        message["image"] = base64_image
    response = client.embeddings.create(
        model="bridgetower-large-itm-mlm-itc", input=[message]
    )
    return response["data"][0]["embedding"]


def load_json_file(file_path):
    # Open the JSON file in read mode
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def display_retrieved_results(results):
    print(f"There is/are {len(results)} retrieved result(s)")
    print()
    for i, res in enumerate(results):
        print(
            f'The caption of the {str(i+1)}-th retrieved result is:\n"{res.page_content}"'
        )
        print()
        try:
            # pylint: disable=undefined-variable
            display(Image.open(res.metadata["metadata"]["extracted_frame_path"]))
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Error: {e} in function: utils.display_retrieved_results", end="\n")
        print("------------------------------------------------------------")


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()

#TODO: sacar este schema de aca
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
        if not isBase64(b64_image):
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
            assert isBase64(image), "image must be under base64 encoding"
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


prediction_guard_llava_conv = Conversation(
    system="",
    roles=("user", "assistant"),
    messages=[],
    version="Prediction Guard LLaVA enpoint Conversation v0",
    sep_style=SeparatorStyle.SINGLE,
    map_roles={"user": "USER", "assistant": "ASSISTANT"},
)

# get PredictionGuard Client


def _getPredictionGuardClient():
    PREDICTION_GUARD_API_KEY = get_prediction_guard_api_key()
    client = PredictionGuard(
        api_key=PREDICTION_GUARD_API_KEY,
        # url=prediction_guard_url_endpoint,# fail to connect
    )
    return client


# helper function to call chat completion endpoint of PredictionGuard given a prompt and an image


# pylint: disable=too-many-arguments
def lvlm_inference(
    prompt,
    image,
    max_tokens: int = 200,
    temperature: float = 0.95,
    top_p: float = 0.1,
    top_k: int = 10,
):
    """
    Perform inference using a language-vision model (LVLM) based on a given prompt and image.

    Args:
        prompt (str):
                The text prompt to guide the inference.
        image (Any):
                The image input to be used in conjunction with the prompt.
        max_tokens (int, optional):
                The maximum number of tokens to generate.
                        Defaults to 200.
        temperature (float, optional):
                The sampling temperature for controlling randomness.
                        Defaults to 0.95.
        top_p (float, optional):
                The cumulative probability for nucleus sampling.
                        Defaults to 0.1.
        top_k (int, optional):
                The number of highest probability vocabulary tokens to keep for top-k filtering.
                        Defaults to 10.

    Returns:
        Any: The result of the inference process.
    """
    # prepare conversation
    conversation = prediction_guard_llava_conv.copy()
    conversation.append_message(conversation.roles[0], [prompt, image])
    return lvlm_inference_with_conversation(
        conversation,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


def lvlm_inference_with_conversation(
    conversation,
    max_tokens: int = 200,
    temperature: float = 0.95,
    top_p: float = 0.1,
    top_k: int = 10,
):
    # get PredictionGuard client
    client = _getPredictionGuardClient()
    # get message from conversation
    messages = conversation.get_message()
    # call chat completion endpoint at Grediction Guard
    response = client.chat.completions.create(
        model="llava-1.5-7b-hf",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return response["choices"][-1]["message"]["content"]
