import sys
import time

from Crypto.Cipher import AES

# Fixed AES-128 key (16 bytes) for fixed-vs-random plaintext TVLA.
AES_KEY = bytes.fromhex(
    "7d2b8f4e0c1a5b9e3f6d8c2a4e7b1f0d"
)
DEFAULT_DURATION_SECONDS = 1.5
BLOCKS_PER_BATCH = 4096


def normalize_plaintext(text: str) -> bytes:

    raw = text.encode("utf-8")

    if len(raw) >= 16:
        return raw[:16]

    return raw + b"\x00" * (16 - len(raw))


def parse_duration(argv: list[str]) -> float:

    if len(argv) < 3:
        return DEFAULT_DURATION_SECONDS

    duration = float(argv[2])

    if duration <= 0:
        raise ValueError("duration must be positive")

    return duration


def run_aes_workload(
    plaintext: bytes,
    duration_seconds: float
) -> int:

    cipher = AES.new(
        AES_KEY,
        AES.MODE_ECB
    )

    buffer = plaintext * BLOCKS_PER_BATCH
    end_time = time.monotonic() + duration_seconds
    checksum = 0

    # Run by wall-clock time, not iteration count, so powermetrics sees a
    # continuously saturated CPU throughout the sampling window.
    while time.monotonic() < end_time:
        buffer = cipher.encrypt(buffer)
        checksum ^= buffer[0]

    return checksum


def main():

    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python3 target.py <plaintext> [duration_seconds]"
        )

    plaintext = normalize_plaintext(sys.argv[1])
    duration_seconds = parse_duration(sys.argv)
    run_aes_workload(
        plaintext,
        duration_seconds
    )


if __name__ == "__main__":
    main()
