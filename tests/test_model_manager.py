import pytest
from word2vec_explorer import ModelManager

def test_model_manager_can_be_created():
    """Test that ModelManager can be instantiated without loading model"""
    manager = ModelManager(auto_load=False)
    assert manager is not None
    assert manager.model is None

def test_word_exists_returns_false_when_not_loaded():
    """Test word_exists returns False when model not loaded"""
    manager = ModelManager(auto_load=False)
    assert manager.word_exists("test") is False
