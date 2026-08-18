from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config.yaml"

__all__ = ["DEFAULT_CONFIG_PATH", "PACKAGE_ROOT"]
