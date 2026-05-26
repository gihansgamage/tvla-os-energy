import sys

from Crypto.Cipher import AES

# Fixed AES-128 key (16 bytes) for fixed-vs-random plaintext TVLA.
AES_KEY = bytes.fromhex(
    "00112233445566778899aabbccddeeff"
)


def normalize_plaintext(text: str) -> bytes:

    raw = text.encode("utf-8")

    if len(raw) >= 16:
        return raw[:16]

    return raw + b"\x00" * (16 - len(raw))


def main():

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 target.py <plaintext>")

    plaintext = normalize_plaintext(sys.argv[1])
    cipher = AES.new(
        AES_KEY,
        AES.MODE_ECB
    )

    block = plaintext

    # Repeat encryption to amplify power signal.
    for _ in range(200000):
        block = cipher.encrypt(block)


if __name__ == "__main__":
    main()