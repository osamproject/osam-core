from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pydantic

from .. import _json
from ._image_embedding import ImageEmbedding
from ._prompt import Prompt


class GenerateRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model: str
    image_embedding: Optional[ImageEmbedding] = pydantic.Field(default=None)
    image: Optional[np.ndarray] = pydantic.Field(default=None)
    prompt: Optional[Prompt] = pydantic.Field(default=None)

    @pydantic.field_validator("image", mode="before")
    @classmethod
    def validate_image(
        cls, image: Optional[Union[str, np.ndarray]]
    ) -> Optional[np.ndarray]:
        if isinstance(image, str):
            return _json.image_b64data_to_ndarray(b64data=image)
        return image


class GenerateResponse(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model: str
    bounding_boxes: List[List[int]]
    image_embedding: Optional[ImageEmbedding] = pydantic.Field(default=None)
    masks: Optional[List[np.ndarray]] = pydantic.Field(default=None)
    texts: Optional[List[str]] = pydantic.Field(default=None)

    @pydantic.field_validator("masks")
    def validate_masks(
        cls, masks: Optional[List[np.ndarray]]
    ) -> Optional[List[np.ndarray]]:
        if masks is None:
            return None

        for mask in masks:
            if mask.dtype != bool:
                raise ValueError("Masks must be boolean arrays")
        return masks

    @pydantic.field_serializer("masks")
    def serialize_masks(self, masks: Optional[List[np.ndarray]]) -> Optional[List[str]]:
        return [
            _json.image_ndarray_to_b64data(ndarray=mask.view(np.uint8) * 255)
            for mask in masks
        ]
