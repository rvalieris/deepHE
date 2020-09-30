

## CNN model training

Train the first step of the MIL task.

to run it needs pytorch and pandas:

`conda install pandas pytorch`

`MIL_train.py` takes as input a tile table and trains the model.

`MIL_inf.py` takes a trained model and outputs a list of the top K tiles per slide.

code was based on: https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019

