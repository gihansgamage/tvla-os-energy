import hashlib
import sys

data = sys.argv[1].encode()

for _ in range(200000):
    hashlib.sha256(data).digest()