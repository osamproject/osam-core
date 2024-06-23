from typing import List
from typing import Type
from typing import Union

import numpy as np
import pydantic


class Prompt(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    points: np.ndarray
    point_labels: np.ndarray

    @pydantic.field_serializer("points")
    def _serialize_points(self: "Prompt", points: np.ndarray) -> List[List[float]]:
        return points.tolist()

    @pydantic.field_serializer("point_labels")
    def _serialize_point_labels(self: "Prompt", point_labels: np.ndarray) -> List[int]:
        return point_labels.tolist()

    @classmethod
    def _validate_points(cls: Type, points: Union[list, np.ndarray]):
        if isinstance(points, list):
            points = np.array(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("points must be 2-dimensional")
        if points.shape[1] != 2:
            raise ValueError("points must have 2 columns")
        return points

    @classmethod
    def _validate_point_labels(cls: Type, point_labels: Union[list, np.ndarray]):
        if isinstance(point_labels, list):
            point_labels = np.array(point_labels, dtype=int)
        if point_labels.ndim != 1:
            raise ValueError("point_labels must be 1-dimensional")
        if not set(np.unique(point_labels).tolist()).issubset({0, 1, 2, 3}):
            raise ValueError("point_labels must contain only 0, 1, 2, or 3")
        return point_labels

    @pydantic.model_validator(mode="after")
    @classmethod
    def _validate_prompt(cls: Type, value: "Prompt"):
        if value.points.shape[0] != value.point_labels.shape[0]:
            raise ValueError(
                "points and point_labels must have the same number of rows"
            )
        return value
