Team members:
Tianming Zhao,
Mingyi Lai,
Sean Chou,
Kelin Ding

## Explaining the problem:
1. According to the crime rate of different communities, increase more police force in the community with high crime rate. logistic regression is used to determine whether the community needs to increase police force and allocate more resources (for example, quality education, incentivized anonymous tip centers, etc.)

2. According to the time range, increase the police force in the peak time of crime, for example, increase the police force at the time when the crime frequency is high in the day and the dark day, and predict the crime frequency through linear regression based on the time line. Then increase the number of police according to the quantity of output

## Classification task
1. logistic regression was used to determine whether a curfew system should be established according to the peak period of historical crimes

## Regression task
* Regression task was used to estimate the number of crimes happend on people with certain sex and age when a certain month and location are specified.
* A simple linear model was implemented as benchmark. The result metric (Mean Absolute Error, MAE) is 0.34. From the trained weights, we separated the inputs into negative and positive related features. For example, in the bar chart we can see longitude is positively-related, while latitude is negative. It matched with the actual distribution of crime rate at Los Angeles: Northeast is safer than Southwest, as the figure shows. We can also find that November and December are months with the fewest crime count and young people are more likely to be victims.
* A 3-layer deep learning model was proposed, built with fully-connected layers, batch normalization layers, and dropout layers. After hyperparameter tuning among batch size, hidden layer dimension, optimizer type, and dropout rate, the result metric for the optimal model is 0.29, which improved 12% in MAE compared with the linear model benchmark.
* Feature importance was calculated based on the trained weights in the proposed deep learning model.

## Future prediction
* The project document also mentioned to use an RNN to discover whether crime is self-exciting. Our work with a simple GRU RNN network shows crime numbers for a certain location (latitude + longitude) and a certain victim sex do have self-exciting property.

## Folder Structure
- /
  - data
    - Crime_Data_from_2020_to_Present.txt
        link to Crime_Data_from_2020_to_Present.csv
    - rawData.txt
        link to rawData.pkl
  - notebooks: original jupyter notebooks
    - dataProcessingAnalysis.ipynb
    - Regression
        - Benchmark
            - Benchmark_Model_Logistic_Regression_(Part_1_Visualization).ipynb
            - Benchmark_Model_Logistic_Regression_(Part_2_Training,_Testing_and_Forecasting).ipynb
            - Benchmarks - Linear Regression.ipynb
        - DeepLearningModel
            - Deep Learning Method - Regression Model.ipynb
            - featureImportance.ipynb 
    - RNN - future prediction
        - Deep Learning Method - RNN Model.ipynb
  - src: conversion to python code
    - regression
        - model
            - benchmark.py: benchmark linear model
            - mlp.py: 3-layer mlp deep regression model
        - utils
            - dataloader.py: dataset loading methods
            - preprocessing.py: data preprocessing utils
        - runners
            - train_benchmark.py: code to train benchmark model
            - train_mlp.py code to train the mlp model
            - usage
            ```sh
            cd src/regression
            python -m runners.train_benchmark
            python -m runners.train_mlp
            ```
    - rnn
        - model
            - RNN.py: a simple GRU network
        - utils
            - dataloader.py: dataset loading methods
            - preprocessing.py: data preprocessing utils
        - runners
            - train_rnn.py: code to train rnn model
            - usage
            ```sh
            cd src/rnn
            python -m runners.train_rnn
            ```
  - README.md

## Google Drive Link

https://drive.google.com/drive/folders/1Rm65cEFUkIiAl-OHQkCh92tewntIA45l?usp=share_link

## License

CC0 1.0 Universal
