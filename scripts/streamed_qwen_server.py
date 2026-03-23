#!/usr/bin/env python3

from streaming_qwen.server import parse_args, run_server


def main() -> None:
    run_server(parse_args())


if __name__ == "__main__":
    main()
