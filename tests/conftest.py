import sys, warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def system():
    from dfmas.system import DarkFactorySystem
    return DarkFactorySystem.build()


@pytest.fixture(scope="session")
def store():
    from dfmas.featurestore import AsOfStore
    return AsOfStore.load()
