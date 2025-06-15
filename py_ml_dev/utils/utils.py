import pickle
from typing import Any

def save_pickle(obj: Any, path: str) -> None:
    """Save an object to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """Load an object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
