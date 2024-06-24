from typing import List
from typing import Optional
from typing import Type

from loguru import logger

from . import types

running_model: Optional[types.Model] = None
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

    response: types.GenerateResponse = running_model.generate(request=request)
    return response
