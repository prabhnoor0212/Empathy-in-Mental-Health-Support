# A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support

This repository is my attempt to reproduce the major results of the paper: [A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support](https://arxiv.org/pdf/2009.08441v1.pdf). 

>📋  Objective

The scope of this repository is limited to the following architecture and tasks proposed in the paper:

Modelling Task:
We have text conversations between seeker (any person who is seeking help with mental health on online platforms such as Reddit) and responder (any person who is volunteering to help a seeker). The goal of this work is to identify empathy in resonse of the responder towards the seeker. The framework of empathic conversations contains three empathy communication mechanisms – Emotional Reactions, Interpretations, and Explorations. For EACH OF THE 3 FRAMEWORKS, the machine learning task is 2 folds:

1) Given a text from seeker, "Identify/Classify" the text response of a responder into 3 levels of communication (No Communication/ Weak Communication/ Strong Communication). 
2) And, extract the rationale, i.e. words from the text that can offer reasonaing behind making the above classification.

NOTE: These 2 tasks need to be performed for all the 3 frameworks of empathy: Emotional Reactions, Interpretations, and Explorations.

This repository provides the implementation for the architecture proposed in the paper for the above mentioned tasks. The code to reproduce the results (classification as well as extraction for 3 said frameworks) as reported by the authors in the paper is also provided. Further, some additional techniques and architectural variations to the authors' work along with results are also provided.

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

NOTE: All default values of hyperparameters and read/write path information as well as brief description is given in config.py file. Please change the dataset path "data_path" depending on the empathy framework dataset and also change the hyper-parameters if needed.

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

>📋  Make sure the trained model weights are present at specified path. Please change the dataset path "data_path" depending on the empathy framework dataset.

## Pre-trained Models

You can download pretrained models for the basic implementation:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

If you want to have a look at the running/testing of all the results, please refer to the [evaluation notebook]().


## Results

The results are as follows:

<img src="https://user-images.githubusercontent.com/43536129/114054591-a03c4280-98ad-11eb-908f-881e4a215e40.jpg">


>📋  Model information

1. "Reported(all data)": Reported in paper (trained on a slightly larger dataset)
2. "Author Run": Results by running author's code (only on the public dataset)
3. "Simple": My implementation Results (only on the public dataset)
4. "Multi-Head": (Stretch Goal) Results with [multi-head attention](https://arxiv.org/pdf/1706.03762.pdf)
5. "Dense-Synthesizer": (Stretch Goal) Results with [Dense-synthesizer](https://openreview.net/pdf?id=H-SPvQtMwm)
6. "Talking Heads": (Stretch Goal) Results with [Talking Heads Attention](https://arxiv.org/pdf/2003.02436.pdf)
7. "ALL": (Stretch Goal) Multi-Head + Dense-Synthesizer + Talking Heads


If you want to have a look at the running/testing of all the results, please refer to the [evaluation notebook]().


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 