"""Command-line interface to TerraTorch."""

from terratorch.cli_tools import build_lightning_cli


def main():
    _ = build_lightning_cli()


if __name__ == "__main__":
    main()
