# Neural_network_xG_model
The following repository stores syntax related to football analytics.

## Description
This repository contains syntax based on David Sumpter's Expected goals neural network modelling tutorial- found here https://soccermatics.readthedocs.io/en/latest/gallery/lesson7/plot_xG_tracking.html.
It creates an xG based on a 3 layer (2 RelU, 1 sigmoid) neural netwrok to predict whether a goal is scored (1) or a goal is not scored (0) based on a basic set of variables obtained from tracking data.

The model itself is based from Indian Super League data and combines both event and tracking data. From the calibration curve (see model assessment viz) we can see that our model sees fluctuations in how it values higher probability goal scoring opportunites. However, we have an AUC of 0.75 and a brier score of 0.08 meaning that the model is acceptable when you take into account the small amount of data it is provided.

The model is then used to calculate open play xG in Euro 2020. This model can be used to identify player performance and can be used as a player evaluation model when considering player recruitment identification.

### statsbomb_jm
This file contains basic code used to open event and tracking data. It also filters the data down to seasonal data and shot data.

### model_var_setup and model_setup_tests
model_var-setup.py creates the features used as independent variables within the neural network model and is associated with model_setup_tests.py which use pytests to test whether the functions work.

### model
This file contains the neural network used to model xG.

### model_training
This file contains code training the neural network to predict xG on Indian Super League event and tracking data. It also includes visualisations of model accuracy (see below):

**Training history**

![Accuracy](https://github.com/Jmann777/Neural_network_xG_model/assets/87671742/911d6b8d-fee2-4fd1-8884-a13e5ee48411)

**Model Assessment**

![Curves](https://github.com/Jmann777/Neural_network_xG_model/assets/87671742/4727bdc6-7089-4774-9943-09f7d4349eed)

### euro_results
This file contains code that applies the neural network created in model.py and trained in model_training.py to Euro 2020 data.

### euro_viz
This file creates visualisations of the reuslts provided in euro_results.py. See the visualisations below:

**Top 10 Players by open play xG**

![xG model Euro 2020 results](https://github.com/Jmann777/Neural_network_xG_model/assets/87671742/54839ddf-2b68-4db0-a4e1-095545d48941)

**Top 10 overperforming players by open play xG per 90** 
(note the sample size for this visualisation was reduced due to minutes played data being webscraped from the FbRef website. The difference in player naming conventions between the statsbomb and FbRef data means some players were excluded)

![xG model Euro 2020 Scatter](https://github.com/Jmann777/Neural_network_xG_model/assets/87671742/b2d19c1d-96a0-4f10-b477-01be87b73674)

### pizza_plot_viz

This file contains code that builds a player radar based on the percentile ranking of finishing and playmaking statistics in Euro 2020. Specifically it examines Kasper Dolberg who has been previously identified as someone who is overperforming his xG (seen in the scatter gram above).

**Kasper Dolberg player radar**

![Kasper Dolberg Euro 2020](https://github.com/Jmann777/Neural_network_xG_model/assets/87671742/2ecc9e96-1834-4041-b101-c7f4d12b13ea)

## Sources
The following sources were used to create this project: <br>
Cavus, M. and Biecek, P., 2022, October. Explainable expected goal models for performance analysis in football analytics. In 2022 IEEE 9th International Conference on Data Science and Advanced Analytics (DSAA) (pp. 1-9). IEEE. <br>
Fernandez, J., Bornn, L. and Cervone, D. (2021) ‘A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions’, Machine Learning, 110, pp. 1389–1427. <br>
Singh, K. (2018) Introducing Expected Threat (xT). Available at: https://karun.in/blog/expected-threat.html (Accessed: 19 March 2024). <br>
Spearman, W., 2018, February. Beyond expected goals. In Proceedings of the 12th MIT sloan sports analytics conference (pp. 1-17). <br>
Sumpter, D. (2022) Expected goals including player positions, Expected Goals including player positions - Soccermatics documentation. Available at: https://soccermatics.readthedocs.io/en/latest/gallery/lesson7/plot_xG_tracking.html (Accessed: 19 March 2024). <br>
Umami, I., Gautama, D.H. and Hatta, H.R., 2021. implementing the Expected Goal (xG) model to predict scores in soccer matches. International Journal of Informatics and Information Systems, 4(1), pp.38-54. <br>
