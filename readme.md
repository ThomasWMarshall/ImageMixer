Citation: I began with the following PyTorch example as a reference implementation of a VAE: https://github.com/pytorch/examples/blob/master/vae/main.py


To use the image mixer on a trained model to mix `A.png` and `B.png`, use `mix.py` as follows:

`python3 mix.py --model pretrained.pt --images A.png B.png`

This will show the result using matplotlib and save images to the `result` directory.


To train a new model, use `train.py` as follows:

`python3 train.py --dataset DATASET_FOLDER --epochs EPOCHS`

This will periodically save out the best model acording to the testing loss to `model.pt`. It will halt when the testing loss stops falling or when the specified number of epochs is reached.
