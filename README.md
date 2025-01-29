# MaCBF for SMGs

### Disclaimers
- This is a rewritten version of [macbf](https://github.com/MIT-REALM/macbf) using PyTorch as opposed to Tensorflow. 
- MaCBF has not successfully been trained on SMGs yet. Exploration of different loss functions or training methods should be done. 

## `train.py`
Trains a base macbf model using the original 2D "cars" simulation from the [macbf](https://github.com/MIT-REALM/macbf) paper. Please refer to `config.py` and `core.py`.

## `evaluate.py`
Evaluates the base macbf model using the original 2D "cars" simulation from the  [macbf](https://github.com/MIT-REALM/macbf). Use the `--vis 1` arguement to visualize. 

## `train_smg.py`
Can be used to finetune or train a new macbf model on a doorway smg scenario. The number of agents and rotation of the environment are augmented to create more robust models. Please refer to `config.py`,  `core.py`, and the parameters in `train_smg.py`.

## `evaluate_smg.py`
Evaluates the macbf model on a doorway smg scenario. An animation of the evaluation is saved to the `./animations/` directory (Use the `--vis 1` arguement to visualize.). Please refer to `config.py`,  `core.py`, and the parameters in `evaluate_smg.py`.

## `env_control.py`
Allows for keyboard control of an agent in the smg environment. Control is [ax, ay] using the arrow keys. 

