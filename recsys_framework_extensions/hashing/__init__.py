import hashlib


def compute_sha256_hash_from_object_repr(
    obj: object,
) -> str:
    encoded_byte_str: bytes = obj.__repr__().encode("utf-8", errors="strict")

    sha256_hasher = hashlib.sha256(
        encoded_byte_str,
        usedforsecurity=False,
    )

    return sha256_hasher.hexdigest()
