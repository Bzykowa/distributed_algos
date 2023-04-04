from multiset import *
import hashlib

HASH_FUNS = {
    "awful": AwfulHash,
    "blake2b": hashlib.blake2b,
    "md5": hashlib.md5,
    "sha1": hashlib.sha1,
    "sha256": hashlib.sha256
}


class AwfulHash:
    def __init__(self, data):
        self.data = data

    def hexdigest(self):
        hash_value = 0
        for byte in self.data:
            hash_value += byte
            hash_value = hash_value % 2 ** 256
        hash_value += 2 ** 8
        return format(hash_value, "0x")


class FloatHash:
    def __init__(self, bit_length: int, function_name: str) -> None:
        self.B = bit_length
        self.h = HASH_FUNS[function_name]

    def __call__(self, data: bytes) -> float:
        hash = self.h(data).hexdigest()
        # Convert to binary, delete "0b" and cut to B length
        binary_form = bin(int(hash, 16))[:self.B]
        return int(binary_form, 2) / (2 ** (len(binary_form)))


def minCount(k: int, h, M_set: Multiset) -> int:
    """
    Calculates expected value of distinct elements in a multiset M.
    """
    M = [1 for _ in range(k)]

    for s in iter(M_set):
        h_s = h(s)
        if h_s not in M and h_s < M[k-1]:
            M[k-1] = h_s
            M.sort()

    if M[k-1] == 1:
        # number of elements different than 1
        return len(M) - M.count(1)
    else:
        return (k-1)/M[k-1]
    
"""
class BadHash:
    def __init__(self, data):
        self.data = data

    def hexdigest(self):
        hash_value = 0
        for byte in self.data:
            hash_value += byte
            hash_value = hash_value % 2 ** 256
        hash_value = (hash_value + 2 ** 8) % 2 ** 256
        hash = bytes(int(format(hash_value, '02b').ljust(256, "0"),2))
        return format(hash, '32x')
        
hash = BadHash("fas".encode()).hexdigest()
# Convert to binary, delete "0b" and cut to B length
binary_form = bin(int(hash, 16))
print(len(binary_form))
"""

