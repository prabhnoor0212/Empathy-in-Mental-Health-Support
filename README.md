# A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support

This repository is my attempt to reproduce the major results of the paper: [A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support](https://arxiv.org/pdf/2009.08441v1.pdf). 

## Objective

>📋 The scope of this repository is limited to reproducing results from the bi-encoder model architecture proposed in the paper.

The comparison of results obtained from my implementation is done with:

  • the scores reported in the paper.
  
  • the scores obtained by re-running the authors’ implementation on the public dataset (a subset of the original dataset used in the paper).



## Requirements

To install requirements:

```setup
pip3 install -r requirements.txt
```

## Training and Evaluation

>📋  To perform training and evaluation in one go, run the command below.

```train
python3 src/run.py --model_write_path="output/out.pth"
```

NOTE: All default values of hyperparameters and read/write path information as well as a brief description are given in the config.py file. Please change the dataset path "data_path" depending on the empathy framework dataset and also change the hyper-parameters if needed.

## Training

To train the model(s) in the paper, run this command:

```train
python3 src/train.py --model_write_path="output/out.pth"
```

>📋  Please change the dataset path "data_path" depending on the empathy framework dataset and also change the hyper-parameters if needed.

## Evaluation

To evaluate results for a trained model, run this command:

```eval
python3 src/eval.py --model_path="output/out.pth"
```

>📋  Make sure the trained model weights are present at the specified path. Please change the dataset path "data_path" depending on the empathy framework dataset.


## Results

The results are as follows:

<img src="https://user-images.githubusercontent.com/43536129/115950516-45375c00-a4f9-11eb-9659-2d51cec0abb6.jpg">


>📋  Model information

1. "Reported(all data)": Reported in the paper (trained on a slightly larger dataset)
2. "Author Run": Results by running author's code (only on the public dataset)
3. "Simple": My implementation Results (only on the public dataset)
4. "Multi-Head": (Stretch Goal) Results with [multi-head attention](https://arxiv.org/pdf/1706.03762.pdf)
5. "Dense-Synthesizer": (Stretch Goal) Results with [Dense-synthesizer](https://openreview.net/pdf?id=H-SPvQtMwm)
6. "Talking Heads": (Stretch Goal) Results with [Talking Heads Attention](https://arxiv.org/pdf/2003.02436.pdf)
7. "ALL": (Stretch Goal) Multi-Head + Dense-Synthesizer + Talking Heads


If you want to have a look at the running/evaluations of all the results, please refer to the [evaluation notebook](https://github.com/prabhnoor0212/Empathy-in-Mental-Health-Support/blob/main/Experiments/Empathy_experiments.ipynb).

## Model Weights & Parameter Configurations files

>📋  As there are 3 mechanisms for the framework of empathy, each link below has 3 config files and 3 corresponding model weights (one for each: Explorations, Emotion Reactions, and Interpretations).

- "Simple": My implementation Results
    - Model Weights & Configs: https://drive.google.com/drive/folders/1vkF3U0DqXSurTnPflWoNEQjwJGPLA6hE?usp=sharing

- "Multi-Head": (Stretch Goal)
    - Model Weights & Configs: https://drive.google.com/drive/folders/1-2_qGujJnAcN2KFfNf8dc5hfds07uanV?usp=sharing

- "Dense-Synthesizer": (Stretch Goal)
    - Model Weights & Configs: https://drive.google.com/drive/folders/1-7bQD1inyi-hzi7n5Gu9VWbWI7tLAZHo?usp=sharing

- "Talking Heads": (Stretch Goal)
    - Model Weights & Configs: https://drive.google.com/drive/folders/1-9GAU2fNl2aqXkLOPLWDMwawx-9M5TEg?usp=sharing

- "ALL": (Stretch Goal) Multi-Head + Dense-Synthesizer + Talking Heads
    - Model Weights & Configs: https://drive.google.com/drive/folders/1-AsdMiXCD4F41cDLboBme0L89WfoRsuV?usp=sharing


## Code Organization Overview

- "Experiments": This directory has the notebook for evaluation of results.
- "TEST": This directory has the code for unit tests.
- "datasets": This directory has all the necessary data files. (Also one example of the processed file is also provided).
- "src": This directory has all the code for the reproducibility project. The sub-directories are:
    - "data_utils": This has the code for data loaders and pre-processors.
    - "models": This has code for all architecture and model building.
    - "pre_trained_modeling": This has all the code necessary for initializing the ROBERTA model used in the final architecture. All the code that has been re-used is given in        this directory
    - "utils": This has the evaluation and training utilities.
    - "eval.py": Code for evaluation (to be used only after a model is trained).
    - "train.py": Code for training the model.
    - "run.py": Code for training and Evaluation in one go.


## Contributing

>📋  MIT LICENSE
