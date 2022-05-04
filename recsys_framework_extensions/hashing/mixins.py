import attrs

from recsys_framework_extensions.hashing import compute_sha256_hash_from_object_repr


@attrs.define(frozen=True)
class MixinSHA256Hash:
    @property
    def sha256_hash(self):
        return compute_sha256_hash_from_object_repr(self)
