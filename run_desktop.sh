#!/bin/bash

# 1. Activate the virtual environment
source "/home/rpicoffee/Thesis/bin/activate"

# 2. Access the Desktop folder
cd "/home/rpicoffee/Desktop/"

# 3. Run main.py
python3 main.py "$@"

# 4. Deactivate the environment
deactivate
