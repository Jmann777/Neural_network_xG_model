# Neural_network_xG_model
The following repository stores syntax related to football analytics.

## Description
This repository contains syntax based on David Sumpter's Expected goals neural network modelling tutorial- found here https://soccermatics.readthedocs.io/en/latest/gallery/lesson7/plot_xG_tracking.html.
It creates an xG based on a 3 layer (2 RelU, 1 sigmoid) neural netwrok to predict whether a goal is scored (1) or a goal is not scored (0) based on a basic set of variables obtained from tracking data.

The model itself is based from Indian Super League data and combines both event and tracking data. The model is then used to calculate open play xG in Euro 2020. This model can be used to identify player performance and can be used as a player evaluation model when considering player recruitment identification.


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
This file contains code that calculates xG in Euro 2020 based on the model and outputs the top 10 players with the highest xG in Euro 2020 according to the model. The results of the model use on the Euro 2020 dataset can be vizualised below:

**Top 10 Players by open play xG**

![xG model Euro 2020 results](https://github.com/Jmann777/Neural_network_xG_model/assets/87671742/4fa14f33-07dc-46b5-9732-e7a6f6f4964d)

**Top 10 overperforming players by open play xG per 90** 
(note the sample size for this visualisation was reduced due to minutes played data being webscraped from the FbRef website. The difference in player naming conventions between the statsbomb and FbRef data means some players were excluded)

![xG model Euro 2020 Scatter](https://github.com/Jmann777/Neural_network_xG_model/assets/87671742/7fdfaadd-631e-4168-bb42-337e2836a141)

## Pizza_plot_viz

This file contains code that builds a player radar based on the percentile ranking of finishing and playmaking statistics in Euro 2020. Specifically it examines Kasper Dolberg who has been previously identified as someone who is overperforming his xG (seen in the scatter gram above).

**Kasper Dolberg player radar**

![Kasper Dolberg Euro 2020](https://github.com/Jmann777/Neural_network_xG_model/assets/87671742/2ecc9e96-1834-4041-b101-c7f4d12b13ea)
