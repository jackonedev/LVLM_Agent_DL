# Multimodal RAG: Chat with Videos

### Part 2: Multimodal Embeddings

Let's imagine an (image-text) pair. This multimodal data pair can be processed by a multimodal embedding model. The model used during the lesson is called BridgeTower, and it generates an embedding of the image-text data pair into a 512-dimensional vector within a multimodal semantic space.

"In this way, a multimodal semantic space is obtained, where the multimodal data pair enhances the closeness of similar vectors in their vector representations, providing an additional level of representational power."

- In the notebook presented here, the **PredictionGuard** "bridgetower-large-itm-mlm-itc" model is used: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1K5Y1NXv-9Z-l-q83y8kjd5JLLb6kbVNv/view?usp=sharing)

- In the notebook presented here, the **CLIP** "ViT-B/32" model is used: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/136rTHsohoFjTX_Lq6INhKHJEALpGz850/view?usp=sharing)


_The model that obtained the most significant results was the BridgeTower from PredictionGuard_



### Part 3: Video Transcription Model

Case 1: The video comes with a transcription file <br />
Case 2: We use the Whisper model for video transcription<br />
Case 3: Video without language solved by caption generation using LlaVA LVLM  (Large Vision Language Model)<br />

The whisper model requires the command-line tool ffmpeg to be installed on your system.

```raw
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

The 3 mentioned cases are presented in the notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/14piuUw1xi0XwQLVAQCudegP3bqu9nfkk/view?usp=sharing)


### Part 4: Multimodal Retriever from LanceDB vector database

First, we will populate data into LanceDB, the vector store consists of video frames + its associated captions. The multimodal RAG is created using LangChain. The metadata is augmented to update the video frame context. So, we increase the transcription segment for more context (because some chunk transcriptions could be short or with lack of meaning).

Presentation Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1c9UuvL-BDBn1vU0ZnGNcC3HEw7iHYX46/view?usp=sharing)


### Part 5: LlaVA: Open-Source Large Vision Language Model

This notebook shows how a multimodal embedding model with a data-pair image-text can be used to generate a multimodal semantic space. Then, the LlaVA model can be set up for Question Answering if we provide an image with a simple description. The examples in the notebook show the capability of the model to pay attention to details based on the context of the description: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1acScV4jpsJoUcL8coO-OFThwIZL__VVS/view?usp=sharing)

_https://huggingface.co/BridgeTower/bridgetower-large-itm-mlm-itc_


### Part 6: Multimodal RAG (MM-RAG) using Langchain

After preprocessing the video data to make it suitable for computing multimodal embeddings using the BridgeTower model, we have ingested our entire video corpus into a multimodal vector store using LanceDB. Now we are going to implement the LVLM called LlaVA (that can take as input both, images and text) and connect it to the LanceDB vector store to create a multimodal RAG system that will allow us to chat with videos. The notebook is available here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1xl8zow9R6oG2NGbcFX2h_-DPeufVsBwg/view?usp=sharing)

