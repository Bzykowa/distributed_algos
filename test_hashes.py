import hashlib
import numpy as np


class AwfulHash:
    """A bad hash for task 6."""

    def __init__(self, data):
        self.data = data

    def __bitstringdigest(self) -> str:
        hash_value = 0
        for byte in self.data:
            hash_value += byte
            hash_value = hash_value
        hash_value = (hash_value + 2 ** 8) % 2 ** 256
        # Sum of bytes + 2^8, padded with 01, cut to 256 len and reversed
        return (format(hash_value, '02b')+"01"*128)[:256][::-1]

    def digest(self) -> bytes:
        s = self.__bitstringdigest()
        return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')

    def hexdigest(self) -> str:
        s = self.__bitstringdigest()
        hash = int(s, 2)
        return hex(hash)


# hash functions for test purposes
HASH_FUNS = {
    "awful": AwfulHash,
    "blake2b": hashlib.blake2b,
    "md5": hashlib.md5,
    "sha1": hashlib.sha1,
    "sha256": hashlib.sha256
}


class FloatHash:
    """
    Class casting a hash function to a float.
    """

    def __init__(self, bit_length: int, function_name: str) -> None:
        # optimal B = 2log2(n) (Birthday paradox)
        self.B = bit_length
        self.h = HASH_FUNS[function_name]

    def __call__(self, data: bytes) -> float:
        hehe = self.h(data).hexdigest()
        # Convert to binary, delete "0b" and cut to B length
        binary_form = bin(int(hehe, 16))[3:]
        binary_form = binary_form[:self.B]
        return int(binary_form, 2) / (2 ** len(binary_form))


class Hash32Bit:
    """Class that returns hash function output cut to 32 bits as binary string."""

    def __init__(self, function_name: str) -> None:
        self.h = HASH_FUNS[function_name]

    def __call__(self, data: bytes) -> str:
        output = self.h(data).hexdigest()
        binary = bin(int(output,16))[3:]
        return binary[:32]


def testhash():
    # h_len = [8, 16, 32, 64, 128]
    h_fun = ["awful", "blake2b", "md5", "sha1", "sha256"]
    for fun in h_fun:
        results = []
        for i in range(10**4):
            hash = FloatHash(160, fun)
            results.append(hash(str(i).encode()))
        print(f"{fun} mean = {np.mean(results)}")
        print(f"{fun} var = {np.var(results)}")
