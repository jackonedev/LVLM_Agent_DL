# Multimodal RAG: Chat with Videos

### Part 1: Multimodal Embeddings

Let's imagine an (image-text) pair. This multimodal data pair can be processed by a multimodal embedding model. The model used during the lesson is called BridgeTower, and it generates an embedding of the image-text data pair into a 512-dimensional vector within a multimodal semantic space.

"In this way, a multimodal semantic space is obtained, where the multimodal data pair enhances the closeness of similar vectors in their vector representations, providing an additional level of representational power."

- In the notebook presented here, the **PredictionGuard** "bridgetower-large-itm-mlm-itc" model is used: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1K5Y1NXv-9Z-l-q83y8kjd5JLLb6kbVNv/view?usp=sharing)

- In the notebook presented here, the **CLIP** "ViT-B/32" model is used: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/136rTHsohoFjTX_Lq6INhKHJEALpGz850/view?usp=sharing)


_The model that obtained the most significant results was the BridgeTower from PredictionGuard_



