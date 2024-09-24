"""
Este modulo se encarga de transformar texto e imagen a embeddings.
Si se proporciona un texto y una imagen, se suman los embeddings
de texto e imagen para conservar la dimension.

Nota:
    - Se utiliza el modelo CLIP de OpenAI para obtener ambos embeddings.
    - Existe una alternativa que podría mejorar la calidad de los embeddings combinados:
        - Modelos de embeddings de texto por Sentence Transformers.
"""

import pathlib
from typing import Union

import clip
import numpy as np
import torch
from PIL import Image

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


def clip_embedding(prompt: str = None, pil_img: Image.Image = None) -> tuple:

    if pil_img and prompt:
        text = clip.tokenize([prompt]).to(device)
        image = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            text_embedding = clip_model.encode_text(text)
            image_embedding = clip_model.encode_image(image)
        return text_embedding.cpu(), image_embedding.cpu()

    elif prompt:
        text = clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            text_embedding = clip_model.encode_text(text)
        return (text_embedding.cpu(),)

    elif pil_img:
        image = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = clip_model.encode_image(image)
        return (image_embedding.cpu(),)

    raise ValueError("You must provide a prompt or an image path.")


def combine_embeddings_concat(
    text_embedding: torch.Tensor,
    image_embedding: torch.Tensor,
) -> torch.Tensor:

    # Concatenar los embeddings de imagen y texto
    combined_embedding = torch.cat((image_embedding, text_embedding), dim=1)
    return combined_embedding


def combine_embeddings_sum(
    text_embedding: torch.Tensor,
    image_embedding: torch.Tensor,
) -> torch.Tensor:

    # Sumar los embeddings de imagen y texto
    combined_embedding = image_embedding + text_embedding
    return combined_embedding


def vit_embedding_from_clip_by_openai(
    prompt: str = None, image_data: Union[str, Image.Image] = None
) -> np.ndarray:

    if isinstance(image_data, str) and pathlib.Path(image_data).exists():
        image_data = Image.open(image_data)

    if prompt and image_data:
        text_embedding, image_embedding = clip_embedding(
            prompt=prompt, pil_img=image_data
        )
        combined_embedding = combine_embeddings_sum(text_embedding, image_embedding)
        return combined_embedding.detach().numpy()[0]

    elif prompt:
        text_embedding = clip_embedding(prompt=prompt)[0]
        return text_embedding.detach().numpy()[0]

    elif image_data:
        image_embedding = clip_embedding(pil_img=image_data)[0]
        return image_embedding.detach().numpy()[0]

    assert (
        image_data is None or pathlib.Path(image_data).exists()
    ), "Image path does not exist."

    raise ValueError("You must provide a prompt or an image path.")
