from . import apis
from . import types


def test_register_model_type():
    class TestModel(types.ModelBase):
        name = "test:latest"

    apis.register_model_type(TestModel)
    assert TestModel in apis.registered_model_types

    assert apis.get_model_type_by_name(name="test") == TestModel
