

## RNN model training

Train the second step of the MIL task.

to run it needs pytorch and pandas:

`conda install pandas pytorch`

`RNN_train.py` takes as input a tile table and trains the model.

`RNN_inf.py` takes a trained model and outputs a list of slides with their predictions.

code was based on: https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019

