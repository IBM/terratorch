#!/bin/bash
pip install --upgrade pip
pip install --user git+https://github.com/The-AI-Alliance/GEO-Bench-2.git@main
pip install --user -e .[wxc,test]

pytest -s --cov=terratorch -v --cov-report term-missing tests


