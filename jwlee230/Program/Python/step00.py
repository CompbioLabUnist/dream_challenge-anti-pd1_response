"""
step00: for general purpose within all step
"""
import hashlib
import hmac
import os
import pickle
import tarfile
import tempfile
import typing

key = bytes("asdf", "UTF-8")


def file_list(path: str) -> typing.List[str]:
    """
    file_list: return a list of files in path
    """
    return list(filter(lambda x: os.path.isfile(x), list(map(lambda x: os.path.join(path, x), os.listdir(path)))))


def directory_list(path: str) -> typing.List[str]:
    """
    directory_list: return a list of directories in path
    """
    return list(filter(lambda x: os.path.isdir(x), list(map(lambda x: os.path.join(path, x), os.listdir(path)))))


def make_hmac(message: bytes) -> bytes:
    """
    make_hmac: return a HMAC
    """
    return hmac.new(key, message, hashlib.sha512).digest()


def make_pickle(path: str, data: typing.Any) -> None:
    """
    make_pickle: create a pickle
    """
    if not path.endswith(".tar.gz"):
        raise ValueError("Path must end with .tar.gz")

    pkl = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    key = make_hmac(pkl)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "data.pkl"), "wb") as f:
            f.write(pkl)
        with open(os.path.join(tmp_dir, "key.txt"), "wb") as f:
            f.write(key)

        with tarfile.open(path, "w:gz") as tar:
            tar.add(os.path.join(tmp_dir, "data.pkl"), arcname="data.pkl")
            tar.add(os.path.join(tmp_dir, "key.txt"), arcname="key.txt")


def read_pickle(path: str) -> typing.Any:
    """
    read_pickle: read a pickle file
    """
    if not path.endswith(".tar.gz"):
        raise ValueError("Path must end with .tar.gz")
    if not tarfile.is_tarfile(path):
        raise ValueError("Path cannot be read as a tar file")

    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(tmp_dir)

        with open(os.path.join(tmp_dir, "data.pkl"), "rb") as f:
            pkl = f.read()
        with open(os.path.join(tmp_dir, "key.txt"), "rb") as f:
            key = f.read()

    if not hmac.compare_digest(make_hmac(pkl), key):
        raise ValueError("Data is not valid")

    return pickle.loads(pkl)
