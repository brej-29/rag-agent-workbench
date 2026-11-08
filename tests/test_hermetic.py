from app.core import config


def test_suite_does_not_read_dotenv():
    assert config._DOTENV_ENABLED is False
    assert config.Settings.model_config["env_file"] is None
