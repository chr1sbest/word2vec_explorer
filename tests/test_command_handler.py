import pytest
from word2vec_explorer import CommandHandler, ModelManager

def test_command_handler_creation():
    """Test CommandHandler can be created with ModelManager"""
    model_manager = ModelManager(auto_load=False)
    handler = CommandHandler(model_manager)
    assert handler is not None
    assert handler.model_manager == model_manager
