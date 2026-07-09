BerryScan_SAHI



folder should contain the following:\
"cbb.pt"\
"BerryScan_SAHI.py"\
"requirements.txt"

Download Source File in this link: https://github.com/BbyZz/BerryScan_SAHI \
Download Trained Model (.pt file) in this link: https://drive.google.com/drive/folders/1VIGT5DD0ncxRiufGtt3mM5hVHgEfcDOz?usp=sharing\
Extract the zip\
In the extracted folder, press the address bar and type “CMD”\
This should open a CMD Termninal\
If you have a virtual environment, activate it. If you don’t have, make one with Anaconda Navigator\
Once your into your virtual environment, type this into the CMD “pip install -r requirements.txt”, this will download all the needed libraries\
Once done with the installation\
Close the cmd terminal, then launch a new one, by doing the step earlier.\
Then activate the environment with the libraries\
Then type “python BerryScan_SAHI.py” to run the InfSlicer program.\

#!/bin/bash

# 1. Activate the virtual environment
source "/home/rpicoffee/Thesis/bin/activate"

# 2. Access the folder
cd "/home/rpicoffee/Downloads/T_Gui"

# 3. Run the application
python3 main.py "$@"

# 4. Deactivate the environment
deactivate
