import dataclasses
import hashlib
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import gdown
import numpy as np
import onnxruntime
import pydantic
from loguru import logger

from . import _contextlib
from . import _json


class ImageEmbedding(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    original_height: int
    original_width: int
    embedding: np.ndarray

    @pydantic.validator("embedding")
    def validate_embedding(cls, embedding):
        if embedding.ndim != 3:
            raise ValueError(
                "embedding must be 3-dimensional: (embedding_dim, height, width)"
            )
        return embedding


class Prompt(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    points: np.ndarray
    point_labels: np.ndarray

    @pydantic.validator("points", pre=True)
    def validate_points(cls, points):
        if isinstance(points, list):
            points = np.array(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("points must be 2-dimensional")
        if points.shape[1] != 2:
            raise ValueError("points must have 2 columns")
        return points

    @pydantic.field_serializer("points")
    def serialize_points(self, points: np.ndarray) -> List[List[float]]:
        return points.tolist()

    @pydantic.validator("point_labels", pre=True)
    def validate_point_labels(cls, point_labels, values):
        if isinstance(point_labels, list):
            point_labels = np.array(point_labels, dtype=int)
        if point_labels.ndim != 1:
            raise ValueError("point_labels must be 1-dimensional")
        if "points" in values and point_labels.shape[0] != values["points"].shape[0]:
            raise ValueError("point_labels must have the same number of rows as points")
        if not set(np.unique(point_labels).tolist()).issubset({0, 1}):
            raise ValueError("point_labels must contain only 0s and 1s")
        return point_labels

    @pydantic.field_serializer("point_labels")
    def serialize_point_labels(self, point_labels: np.ndarray) -> List[int]:
        return point_labels.tolist()


class GenerateRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model: str
    image: np.ndarray
    prompt: Optional[Prompt] = pydantic.Field(default=None)

    @pydantic.validator("image", pre=True)
    def validate_image(cls, image: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image, str):
            return _json.image_b64data_to_ndarray(b64data=image)
        return image


class GenerateResponse(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model: str
    mask: np.ndarray

    @pydantic.field_serializer("mask")
    def serialize_mask(self, mask: np.ndarray) -> str:
        return _json.image_ndarray_to_b64data(ndarray=mask)


@dataclasses.dataclass
class ModelBlob:
    url: str
    hash: str

    @property
    def path(self):
        return os.path.expanduser(f"~/.cache/osam/models/blobs/{self.hash}")

    @property
    def size(self):
        if os.path.exists(self.path):
            return os.stat(self.path).st_size
        else:
            return None

    @property
    def modified_at(self):
        if os.path.exists(self.path):
            return os.stat(self.path).st_mtime
        else:
            return None

    def pull(self):
        gdown.cached_download(url=self.url, path=self.path, hash=self.hash)

    def remove(self):
        if os.path.exists(self.path):
            logger.debug("Removing model blob {path!r}", path=self.path)
            os.remove(self.path)
        else:
            logger.warning("Model blob {path!r} not found", path=self.path)


class ModelBase:
    name: str

    _blobs: Dict[str, ModelBlob]
    _inference_sessions: Dict[str, onnxruntime.InferenceSession]

    def __init__(self):
        self.pull()

        providers = None
        self._inference_sessions = {}
        for key, blob in self._blobs.items():
            try:
                # Try to use all of the available providers e.g., cuda, tensorrt.
                if providers is None:
                    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    else:
                        providers = ["CPUExecutionProvider"]
                # Suppress all the error messages from the missing providers.
                with _contextlib.suppress():
                    inference_session = onnxruntime.InferenceSession(
                        blob.path, providers=providers
                    )
            except Exception as e:
                # Even though there is fallback in onnxruntime, it won't always work.
                # e.g., CUDA is installed and CUDA_PATH is set, but CUDA_VISIBLE_DEVICES
                # is empty. We fallback to cpu in such cases.
                logger.error(
                    "Failed to create inference session with providers {providers!r}. "
                    "Falling back to ['CPUExecutionProvider']",
                    providers=providers,
                    e=e,
                )
                providers = ["CPUExecutionProvider"]
                inference_session = onnxruntime.InferenceSession(
                    blob.path, providers=providers
                )
            self._inference_sessions[key] = inference_session

            providers = inference_session.get_providers()
        logger.info(
            "Initialized inference sessions with providers {providers!r}",
            providers=providers,
        )

    @classmethod
    def pull(cls):
        for blob in cls._blobs.values():
            blob.pull()

    @classmethod
    def remove(cls):
        for blob in cls._blobs.values():
            blob.remove()

    @classmethod
    def get_id(cls) -> str:
        return hashlib.md5(
            "+".join(blob.hash for blob in cls._blobs.values()).encode()
        ).hexdigest()[:12]

    @classmethod
    def get_size(cls) -> Optional[int]:
        size = 0
        for blob in cls._blobs.values():
            if blob.size is None:
                return None
            size += blob.size
        return size

    @classmethod
    def get_modified_at(cls) -> Optional[float]:
        modified_at = 0
        for blob in cls._blobs.values():
            if blob.modified_at is None:
                return None
            modified_at = max(modified_at, blob.modified_at)
        return modified_at

    def encode_image(self, image: np.ndarray) -> ImageEmbedding:
        raise NotImplementedError

    def generate_mask(
        self, image_embedding: ImageEmbedding, prompt: Prompt
    ) -> np.ndarray:
        raise NotImplementedError
