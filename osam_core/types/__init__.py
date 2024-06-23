import warnings

from ._blob import Blob  # noqa: F401
from ._generate import GenerateRequest  # noqa: F401
from ._generate import GenerateResponse  # noqa: F401
from ._image_embedding import ImageEmbedding  # noqa: F401
from ._model import Model  # noqa: F401
from ._prompt import Prompt  # noqa: F401


class ModelBase(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warning(
            "This class is deprecated, use `Model` instead.", DeprecationWarning
        )


class ModelBlob(Blob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warning(
            "This class is deprecated, use `Blob` instead.", DeprecationWarning
        )
