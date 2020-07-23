import pytest

from models import models_manager


def test_models_manager_get_model_unsupported():
    models_mngr = models_manager.Models()
    model_name = ""

    with pytest.raises(ValueError):
        model = models_mngr.get_model(model_name)
