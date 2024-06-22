from typing import List
from typing import Optional
from typing import Type

import numpy as np
from loguru import logger

from . import types

running_model: Optional[types.ModelBase] = None
registered_model_types: List[Type] = []


def register_model_type(model: Type) -> None:
    global registered_model_types

    for registered_model in registered_model_types:
        if registered_model.name == model.name:
            logger.warning(f"Model {model.name!r} is already registered")
            return
    else:
        registered_model_types.append(model)


def get_model_type_by_name(name: str) -> Type:
    model_name: str
    if ":" in name:
        model_name = name
    else:
        model_name = f"{name}:latest"

    for cls in registered_model_types:
        if cls.name == model_name:
            break
    else:
        raise ValueError(f"Model {name!r} not found.")
    return cls


def generate(request: types.GenerateRequest) -> types.GenerateResponse:
    global running_model

    model_type = get_model_type_by_name(name=request.model)
    if running_model is None or running_model.name != model_type.name:
        running_model = model_type()
    assert running_model is not None

    image: np.ndarray = request.image

    if request.prompt is None:
        height, width = image.shape[:2]
        prompt = types.Prompt(
            points=np.array([[width / 2, height / 2]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
        )
        logger.warning(
            "Prompt is not given, so using the center point as prompt: {prompt!r}",
            prompt=prompt,
        )
    else:
        prompt = request.prompt

    image_embedding: types.ImageEmbedding = running_model.encode_image(image=image)
    mask: np.ndarray = running_model.generate_mask(
        image_embedding=image_embedding, prompt=prompt
    )
    return types.GenerateResponse(model=request.model, mask=mask)
