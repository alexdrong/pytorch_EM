import argparse
import logging
import sys


def parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loglevel', type=str, help='log level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    logging.basicConfig(level=args.loglevel)


if __name__ == "__main__":
    main()
