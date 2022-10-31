#!/bin/bash

source venv/bin/activate
python -u marker.py --summary results/epoch_0/predictions.json --reference results/references.json --source results/source.json --rouge --factcc