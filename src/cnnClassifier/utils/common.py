import os
import json
import base64
import joblib  # type: ignore[import-untyped]
import yaml
from pathlib import Path
from typing import Any, cast
from box import ConfigBox  # type: ignore[import-untyped]
from box.exceptions import BoxValueError  # type: ignore[import-untyped]
from ensure import ensure_annotations  # type: ignore[import-untyped]
from cnnClassifier import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its content as a ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        BoxValueError: If the YAML content is invalid.

    Returns:
        ConfigBox: Parsed YAML content with dot-access support.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError(f"YAML file is empty: {path_to_yaml}")
            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError as e:
        raise BoxValueError(f"Invalid YAML content in {path_to_yaml}: {e}")


def create_directories(path_to_directories: list[Path], verbose: bool = True) -> None:
    """Creates a list of directories if they do not already exist.

    Args:
        path_to_directories (list[Path]): List of directory paths to create.
        verbose (bool): Whether to log each created directory. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(str(path), exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Saves a dictionary as a JSON file.

    Args:
        path (Path): Path where the JSON file will be saved.
        data (dict[str, Any]): Dictionary to save.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON saved to: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads a JSON file and returns its content as a ConfigBox.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: JSON content with dot-access support.
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON loaded from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path) -> None:
    """Saves any Python object as a binary file using joblib.

    Args:
        data (Any): Object to serialize (e.g. model, scaler).
        path (Path): Destination path for the binary file.
    """
    joblib.dump(value=data, filename=path)  # type: ignore[no-untyped-call]
    logger.info(f"Binary file saved to: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads a binary file saved with joblib.

    Args:
        path (Path): Path to the binary file.

    Returns:
        Any: The deserialized Python object.
    """
    data: Any = cast(Any, joblib.load(path))  # type: ignore[no-untyped-call]
    logger.info(f"Binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """Returns the size of a file in kilobytes (KB).

    Args:
        path (Path): Path to the file.

    Returns:
        str: File size as a human-readable string, e.g. "~ 24 KB".
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def decode_image(imgstring: str, file_name: str) -> None:
    """Decodes a base64-encoded image string and writes it to a file.
    Used by the Flask prediction endpoint to receive images via API.

    Args:
        imgstring (str): Base64-encoded image string.
        file_name (str): Destination file path to write the decoded image.
    """
    imgdata = base64.b64decode(imgstring)
    with open(file_name, "wb") as f:
        f.write(imgdata)
    logger.info(f"Image decoded and saved to: {file_name}")


def encode_image_into_base64(image_path: str) -> str:
    """Reads an image file and encodes it into a base64 string.
    Used to return prediction results as base64 over the API.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    logger.info(f"Image encoded to base64 from: {image_path}")
    return encoded
