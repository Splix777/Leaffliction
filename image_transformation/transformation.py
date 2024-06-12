import argparse
import sys


def main():
    if len(sys.argv) <= 2:
        print("General Options: ")
        print("  -h      Show this help message and exit")
        print("  -src    Source image path or directory")
        print("  -dst    Destination image directory (Must be used with -src)")
        return

    parser = argparse.ArgumentParser()

    parser.add_argument("-src", help="Source image path or directory")
    parser.add_argument("-dst", help="Destination image directory")

    args = parser.parse_args()


if __name__ == '__main__':
    main()