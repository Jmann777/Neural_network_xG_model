# Neural_network_xG_model
The following repository stores syntax related to football analytics.

## Description
This repository contains syntax based on David Sumpter's Expected goals neural network modelling tutorial- found here https://soccermatics.readthedocs.io/en/latest/gallery/lesson7/plot_xG_tracking.html.
It creates an xG based on a 3 layer (2 RelU, 1 sigmoid) neural netwrok to predict whether a goal is scored (1) or a goal is not scored (0) based on a basic set of variables obtained from tracking data.

The model itself is based from Indian Super League data and combines both event and tracking data. The model is then used to calculate xG in Euro 2020.


## xG advancement.py
This python file contains the main code needed to create the model. It initially opens the data using the statsbomb module (see below) before creating model variables and the neural network model itself.
The neural network is fitted over 1000 epochs and uses the early stopping callback method. The model is assessed through the use of a validation dataset to investigate accuracy, loss as well as using ROC, and Brier scores.

## Statsbomb.py
This file contains basic code used to open event and tracking data alongside basic mathematical conversions to account for the given lenght and width of a football pitch.

## Shots_Features_Sb.py
This file contains the code related to the calculation of a basic xG model around angle and distance of a given shot from the goal.

## Model
This file contains the neural network used to model xG

## Euro Test
This file contains code that calculates xG in Euro 2020 based on the model and outputs the top 10 players with the highest xG in Euro 2020 accroding to the model.

Ther results are below:
                          player_name    our_xG
0           Álvaro Borja Morata Martín  2.620025
1                           Harry Kane  2.490195
2                          Kai Havertz  2.299895
3  Cristiano Ronaldo dos Santos Aveiro  2.136920
4                      Raheem Sterling  2.041717
5                        Ciro Immobile  1.999741
6                        Karim Benzema  1.893814
7                        Patrik Schick  1.798568
8                   Robert Lewandowski  1.728081
9         Diogo José Teixeira da Silva  1.687015
