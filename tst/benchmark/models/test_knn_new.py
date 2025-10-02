import pytest


def test_knn_new():
    toy_model_params = {}
    model_hyperparameters = toy_model_params

    try:
        from autogluon.tabular.testing import FitHelper
        from tabrepo.benchmark.models.ag.knn.knn_model import KNN_NewModel
        model_cls = KNN_NewModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
