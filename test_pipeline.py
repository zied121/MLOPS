"""
test_pipeline.py
----------------
Unit tests for model_pipeline.py functions.
Run with: pytest test_pipeline.py -v
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_PATH = "Churn_Modelling.csv"


@pytest.fixture
def sample_data():
    """Return a small prepared dataset for testing."""
    return prepare_data(DATA_PATH, test_size=0.2, random_state=42)


@pytest.fixture
def trained_model(sample_data):
    """Return a trained RandomForest model."""
    x_train, x_test, y_train, y_test = sample_data
    return train_model(x_train, y_train, n_estimators=10, random_state=42)


# ---------------------------------------------------------------------------
# Tests – prepare_data
# ---------------------------------------------------------------------------

class TestPrepareData:

    def test_returns_four_splits(self, sample_data):
        """prepare_data must return exactly 4 objects."""
        assert len(sample_data) == 4

    def test_correct_split_sizes(self, sample_data):
        x_train, x_test, y_train, y_test = sample_data
        total = len(x_train) + len(x_test)
        assert len(x_test) == pytest.approx(total * 0.2, abs=5)

    def test_no_missing_values(self, sample_data):
        x_train, x_test, y_train, y_test = sample_data
        assert x_train.isnull().sum().sum() == 0
        assert x_test.isnull().sum().sum() == 0

    def test_dropped_columns_absent(self, sample_data):
        x_train, _, _, _ = sample_data
        forbidden = {"RowNumber", "CustomerId", "Surname", "Geography"}
        assert forbidden.isdisjoint(set(x_train.columns))

    def test_gender_is_numeric(self, sample_data):
        x_train, _, _, _ = sample_data
        assert pd.api.types.is_numeric_dtype(x_train["Gender"])

    def test_target_binary(self, sample_data):
        _, _, y_train, y_test = sample_data
        assert set(y_train.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests – train_model
# ---------------------------------------------------------------------------

class TestTrainModel:

    def test_returns_classifier(self, trained_model):
        assert isinstance(trained_model, RandomForestClassifier)

    def test_model_is_fitted(self, trained_model):
        """A fitted model has the estimators_ attribute."""
        assert hasattr(trained_model, "estimators_")

    def test_predict_shape(self, trained_model, sample_data):
        _, x_test, _, _ = sample_data
        preds = trained_model.predict(x_test)
        assert len(preds) == len(x_test)

    def test_predict_binary_output(self, trained_model, sample_data):
        _, x_test, _, _ = sample_data
        preds = trained_model.predict(x_test)
        assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests – evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:

    def test_returns_dict_with_keys(self, trained_model, sample_data):
        _, x_test, _, y_test = sample_data
        result = evaluate_model(trained_model, x_test, y_test, show_plot=False)
        assert "accuracy" in result
        assert "report" in result
        assert "confusion_matrix" in result

    def test_accuracy_in_range(self, trained_model, sample_data):
        _, x_test, _, y_test = sample_data
        result = evaluate_model(trained_model, x_test, y_test, show_plot=False)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_confusion_matrix_shape(self, trained_model, sample_data):
        _, x_test, _, y_test = sample_data
        result = evaluate_model(trained_model, x_test, y_test, show_plot=False)
        assert result["confusion_matrix"].shape == (2, 2)


# ---------------------------------------------------------------------------
# Tests – save_model / load_model
# ---------------------------------------------------------------------------

class TestSaveLoadModel:

    def test_save_creates_file(self, trained_model):
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        try:
            save_model(trained_model, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_load_returns_classifier(self, trained_model):
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        try:
            save_model(trained_model, path)
            loaded = load_model(path)
            assert isinstance(loaded, RandomForestClassifier)
        finally:
            os.unlink(path)

    def test_loaded_model_predicts_same(self, trained_model, sample_data):
        _, x_test, _, _ = sample_data
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        try:
            save_model(trained_model, path)
            loaded = load_model(path)
            original_preds = trained_model.predict(x_test)
            loaded_preds   = loaded.predict(x_test)
            np.testing.assert_array_equal(original_preds, loaded_preds)
        finally:
            os.unlink(path)
