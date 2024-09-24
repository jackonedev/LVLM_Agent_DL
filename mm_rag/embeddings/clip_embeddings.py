import numpy as np
import torch
from PIL import Image
import clip


# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


def clip_embedding(prompt: str, image_path: str = None) -> list:
    text = clip.tokenize([prompt]).to(device)
    text_embedding = clip_model.encode_text(text)

    if image_path:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = clip_model.encode_image(image)
        
        return text_embedding, image_embedding.cpu()
    
    return text_embedding


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


def vit_embedding_from_clip_by_openai(prompt: str, image_path: str = None) -> np.ndarray:
    if image_path is not None:
        text_embedding, image_embedding = clip_embedding(prompt, image_path)
        combined_embedding = combine_embeddings_sum(text_embedding, image_embedding)
        return combined_embedding.detach().numpy()[0]
    
    return clip_embedding(prompt).detach().numpy()[0]
