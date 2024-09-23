# pylint: disable=missing-module-docstring
import os
# import time
from pathlib import Path
# pylint: disable=import-error
import gradio as gr
import lancedb
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from moviepy.video.io.VideoFileClip import VideoFileClip

from mm_rag.embeddings.bridgetower_embeddings import BridgeTowerEmbeddings
from mm_rag.MLM.client import PredictionGuardClient
from mm_rag.MLM.lvlm import LVLM
from mm_rag.vectorstores.multimodal_lancedb import MultimodalLanceDB
from utils import (
    lvlm_inference_with_conversation,
    prediction_guard_llava_conv,
    PROMPT_TEMPLATE,
    SERVER_ERROR_MSG,
    LANCEDB_HOST_FILE,
    TBL_NAME,
)
from gradio_schema import GradioInstance, SeparatorStyle
from frontend.css import css
from frontend.html import html_title



def set_default(host_file, tbl_name):
    """
    Set default LanceDB host file and table name
    """
    host_file = "./shared_data/.lancedb"
    tbl_name = "demo_tbl"
    return host_file, tbl_name


# function to split video at a timestamp
# pylint: disable=missing-function-docstring
# pylint: disable=too-many-arguments
def split_video(
    video_path,
    timestamp_in_ms,
    output_video_path: str = "./shared_data/splitted_videos",
    output_video_name: str = "video_tmp.mp4",
    play_before_sec: int = 3,
    play_after_sec: int = 3,
):
    timestamp_in_sec = int(timestamp_in_ms / 1000)
    # create output_video_name folder if not exist:
    Path(output_video_path).mkdir(parents=True, exist_ok=True)
    output_video = os.path.join(output_video_path, output_video_name)
    with VideoFileClip(video_path) as video:
        duration = video.duration
        start_time = max(timestamp_in_sec - play_before_sec, 0)
        end_time = min(timestamp_in_sec + play_after_sec, duration)
        new = video.subclip(start_time, end_time)
        new.write_videofile(output_video, audio_codec="aac")
    return output_video


# define default rag_chain
# pylint: disable=missing-function-docstring
def get_default_rag_chain():
    # declare host file
    # declare table name
    lancedb_host_file , tbl_name = set_default(LANCEDB_HOST_FILE, TBL_NAME)

    # initialize vectorstore
    # pylint: disable=unused-variable
    db = lancedb.connect(lancedb_host_file)

    # initialize an BridgeTower embedder
    embedder = BridgeTowerEmbeddings()

    # Creating a LanceDB vector store
    vectorstore = MultimodalLanceDB(
        uri=lancedb_host_file, embedding=embedder, table_name=tbl_name
    )
    # creating a retriever for the vector store
    retriever_module = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 1}
    )

    # initialize a client as PredictionGuardClien
    client = PredictionGuardClient()
    # initialize LVLM with the given client
    lvlm_inference_module = LVLM(client=client)

    def prompt_processing(chain_input):
        # get the retrieved results and user's query
        retrieved_results = chain_input["retrieved_results"]
        user_query = chain_input["user_query"]
        # get the first retrieved result by default
        retrieved_result = retrieved_results[0]

        # get all metadata of the retrieved video segment
        metadata_retrieved_video_segment = retrieved_result.metadata["metadata"]

        # get the frame and the corresponding transcript, path to extracted frame,
        # path to whole video, and time stamp of the retrieved video segment.
        transcript = metadata_retrieved_video_segment["transcript"]
        frame_path = metadata_retrieved_video_segment["extracted_frame_path"]
        chain_output = {
            "prompt": PROMPT_TEMPLATE.format(
                transcript=transcript, user_query=user_query
            ),
            "image": frame_path,
            "metadata": metadata_retrieved_video_segment,
        }
        return chain_output

    # initialize prompt processing module as a
    # Langchain RunnableLambda of function prompt_processing
    prompt_processing_module = RunnableLambda(prompt_processing)

    # the output of this new chain will be a dictionary
    mm_rag_chain_with_retrieved_image = (
        RunnableParallel(
            {"retrieved_results": retriever_module, "user_query": RunnablePassthrough()}
        )
        | prompt_processing_module
        | RunnableParallel(
            {
                "final_text_output": lvlm_inference_module,
                "input_to_lvlm": RunnablePassthrough(),
            }
        )
    )
    return mm_rag_chain_with_retrieved_image


def get_gradio_instance(mm_rag_chain=None):
    if mm_rag_chain is None:
        mm_rag_chain = get_default_rag_chain()

    instance = GradioInstance(
        system="",
        roles=prediction_guard_llava_conv.roles,
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.SINGLE,
        sep="\n",
        path_to_img=None,
        video_title=None,
        caption=None,
        mm_rag_chain=mm_rag_chain,
    )
    return instance


gr.set_static_paths(paths=["./assets/"])
theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c100="#dbeafe",
        c200="#bfdbfe",
        c300="#93c5fd",
        c400="#60a5fa",
        c50="#eff6ff",
        c500="#0054ae",
        c600="#00377c",
        c700="#00377c",
        c800="#1e40af",
        c900="#1e3a8a",
        c950="#0a0c2b",
    ),
    secondary_hue=gr.themes.Color(
        c100="#dbeafe",
        c200="#bfdbfe",
        c300="#93c5fd",
        c400="#60a5fa",
        c50="#eff6ff",
        c500="#0054ae",
        c600="#0054ae",
        c700="#0054ae",
        c800="#1e40af",
        c900="#1e3a8a",
        c950="#1d3660",
    ),
).set(
    body_background_fill_dark="*primary_950",
    body_text_color_dark="*neutral_300",
    border_color_accent="*primary_700",
    border_color_accent_dark="*neutral_800",
    block_background_fill_dark="*primary_950",
    block_border_width="2px",
    block_border_width_dark="2px",
    button_primary_background_fill_dark="*primary_500",
    button_primary_border_color_dark="*primary_500",
)

dropdown_list = [
    "What is the name of one of the astronauts?",
    "An astronaut's spacewalk",
    "What does the astronaut say?",
]

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

# pylint: disable=unused-argument
def clear_history(state, request: gr.Request):
    state = get_gradio_instance(state.mm_rag_chain)
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 1


# pylint: disable=unused-argument
def add_text(state, text, request: gr.Request):
    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 1

    text = text[:1536]  # Hard cut-off

    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 1


# pylint: disable=unused-argument
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=fixme
# TODO: la funcion tiene muchas variables locales
# TODO: la funcion tiene muchos bloques condicionales
# TODO: la funcion debe ser refactorizada
def http_bot(state, request: gr.Request):
    # start_tstamp = time.time()

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        path_to_sub_videos = state.get_path_to_subvideos()
        yield (state, state.to_gradio_chatbot(), path_to_sub_videos) + (
            no_change_btn,
        ) * 1
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        new_state = get_gradio_instance(state.mm_rag_chain)
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    all_images = state.get_images(return_pil=False)

    # Make requests
    is_very_first_query = True
    if len(all_images) == 0:
        # first query need to do RAG
        # Construct prompt
        prompt_or_conversation = state.get_prompt_for_rag()
    else:
        # subsequence queries, no need to do Retrieval
        is_very_first_query = False
        prompt_or_conversation = state.get_conversation_for_lvlm()

    if is_very_first_query:
        executor = state.mm_rag_chain
    else:
        executor = lvlm_inference_with_conversation

    state.messages[-1][-1] = "▌"
    path_to_sub_videos = state.get_path_to_subvideos()
    yield (state, state.to_gradio_chatbot(), path_to_sub_videos) + (disable_btn,) * 1

    try:
        if is_very_first_query:
            # get response by invoke executor chain
            response = executor.invoke(prompt_or_conversation)
            message = response["final_text_output"]
            if "metadata" in response["input_to_lvlm"]:
                metadata = response["input_to_lvlm"]["metadata"]
                if (
                    state.path_to_img is None
                    and "input_to_lvlm" in response
                    and "image" in response["input_to_lvlm"]
                ):
                    state.path_to_img = response["input_to_lvlm"]["image"]

                if state.path_to_video is None and "video_path" in metadata:
                    video_path = metadata["video_path"]
                    mid_time_ms = metadata["mid_time_ms"]
                    splited_video_path = split_video(video_path, mid_time_ms)
                    state.path_to_video = splited_video_path

                if state.caption is None and "transcript" in metadata:
                    state.caption = metadata["transcript"]
            else:
                raise ValueError("Response's format is changed")
        else:
            # get the response message by directly call PredictionGuardAPI
            message = executor(prompt_or_conversation)
    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(f"Error: {e} en la función gradio_utils.http_bot", end="\n")
        state.messages[-1][-1] = SERVER_ERROR_MSG
        yield (state, state.to_gradio_chatbot(), None) + (enable_btn,)
        return

    state.messages[-1][-1] = message
    path_to_sub_videos = state.get_path_to_subvideos()
    # path_to_image = state.path_to_img
    # caption = state.caption
    # # print(path_to_sub_videos)
    # # print(path_to_image)
    # # print('caption: ', caption)
    yield (state, state.to_gradio_chatbot(), path_to_sub_videos) + (enable_btn,) * 1

    # finish_tstamp = time.time()
    return


def get_demo(rag_chain=None):
    if rag_chain is None:
        rag_chain = get_default_rag_chain()

    with gr.Blocks(theme=theme, css=css) as demo:
        # gr.Markdown(description)
        instance = get_gradio_instance(rag_chain)
        state = gr.State(instance)
        demo.load(
            None,
            None,
            js="""
      () => {
      const params = new URLSearchParams(window.location.search);
      if (!params.has('__theme')) {
        params.set('__theme', 'dark');
        window.location.search = params.toString();
      }
      }""",
        )
        gr.HTML(value=html_title)
        with gr.Row():
            with gr.Column(scale=4):
                video = gr.Video(
                    height=512, width=512, elem_id="video", interactive=False
                )
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="Multimodal RAG Chatbot",
                    height=512,
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        # textbox.render()
                        textbox = gr.Dropdown(
                            dropdown_list,
                            allow_custom_value=True,
                            # show_label=False,
                            # container=False,
                            label="Query",
                            info="Enter your query here or choose a sample from the dropdown list!",
                        )
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(
                            value="Send", variant="primary", interactive=True
                        )
                # pylint: disable=unused-variable
                with gr.Row(elem_id="buttons") as button_row:
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

        btn_list = [clear_btn]

        clear_btn.click(
            clear_history, [state], [state, chatbot, textbox, video] + btn_list
        )
        submit_btn.click(
            add_text,
            [state, textbox],
            [
                state,
                chatbot,
                textbox,
            ]
            + btn_list,
        ).then(
            http_bot,
            [state],
            [state, chatbot, video] + btn_list,
        )
    return demo
