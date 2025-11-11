#!/bin/bash
pip install --upgrade pip
pip install --user -e .[wxc,test,geobenchv2]

pytest -s --cov=terratorch -v --cov-report term-missing tests


