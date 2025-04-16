# Copyright contributors to the Terratorch project

"""Command-line interface to TerraTorch."""

from terratorch.cli_tools import build_lightning_cli
import sys

try:
    from benchmark.main import main as iterate_main

    TERRATORCH_ITERATE_INSTALLED = True

except ImportError:
    TERRATORCH_ITERATE_INSTALLED = False


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "iterate":
        # if user runs "terratorch iterate" and terratorch-iterate has not been installed
        if not TERRATORCH_ITERATE_INSTALLED:
            print(
                (
                    "Error! terratorch-iterate has not been installed. If you want to install it,"
                    "run 'pip install terratorch-iterate'"
                )
            )
        # if user runs "terratorch iterate" and terratorch-iterate has been installed
        else:
            # delete iterate argument
            del sys.argv[1]
            iterate_main()
    else:
        _ = build_lightning_cli()


if __name__ == "__main__":
    main()
