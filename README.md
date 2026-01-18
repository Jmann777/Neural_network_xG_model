# Introduction
The following project evaluates model performance for xG prediction based on a 3 layer neural network (2 RelU, 1 sigmoid). The model is trained on event and tracking data taken from 2021/22 Indian Super League (taken from Statsbomb).

# Setup

### Data Collection

Event and tracking data was taken from the Indian Super League as well as Euro 2020 (provided by Statsbomb). Both datasets were prepped and cleaned with the ISL dataset being used for modelling and the modle then being applied to the Euro 2020 dataset. As well as event and tracking data, minutes played data was used (provided by a kaggle dataset) to standardise xG per 90 when assessing results. 

### Data Preparation + Feature Engineering
To prepare the features for the model a number of transformations were performed to both the event and tracking data. The data engineering to create the iondependent variables included:
- Goal angle + distance calcualtions
- Calculations of the distance between the shot and the goalkeeper
- Calculations to count the number of players with 3 meters of the shot.
- Calculations to determine the number of players within a defined traingle around the shot location.
- The goalkeeper distance to the goal
- Whether the shot was a header or regular

# Modelling

### Metric Selection
As dicussed above the model metrics surrounded basic xG calcualtions (distance + angle, regular shots vs headers) as well as metrics based on tracking data (all other metrics). The concept behind this was to build upon the standard xG calcuation with the addition of tracking calucations which take a step towards increasing the complexity of the model by applying external effects on the ball (opposition and teammates). 

### Model Evaluation + Development
The model was evaluated using training accuracy + loss to assess model training fit (figure 1), as well as ROC + AUC to assess accuracy (figure 2). In addition, the model seeks to optimise its Brier Score using the Adam optimiser, applies a learning rate of 0.001, and sets a patience level of 50.

The model evaluations were visualised and are shown in the figures below. From these figures we can see that the model fits the data well and provides a sufficient predictive capability with the amount of data vaialble for training.

**Figure 1- Training history**

![Accuracy](https://github.com/Jmann777/Neural_network_xG_model/assets/87671742/911d6b8d-fee2-4fd1-8884-a13e5ee48411)

**Figure 2- Model Assessment**

![Curves](https://github.com/Jmann777/Neural_network_xG_model/assets/87671742/4727bdc6-7089-4774-9943-09f7d4349eed)

# Results

### Applying the results to find high performing players
After training and evaluation the model was applied to Euro 2020 data. Once applied, xG was standardised by minutes played through the creation of the xG per 90 variable (minimum of 90 minutes played). Alongside total xG, xG per 90 informed high performing players and was visualsied below. These visualisations included total open play xG and xG overperformance per 90.

**Top 10 Players by open play xG**

![xG model Euro 2020 results](https://github.com/Jmann777/Neural_network_xG_model/assets/87671742/54839ddf-2b68-4db0-a4e1-095545d48941)

**Top 10 overperforming players by open play xG per 90** 
(note the sample size for this visualisation was reduced due to minutes played data being webscraped from the FbRef website. The difference in player naming conventions between the statsbomb and FbRef data means some players were excluded)

![xG model Euro 2020 Scatter](https://github.com/Jmann777/Neural_network_xG_model/assets/87671742/b2d19c1d-96a0-4f10-b477-01be87b73674)

# Conclusions
The neural network xG model produced in this project provided evidence of the combination of event and tracking data and their uses in modelling xG. As the dataset used to train the model was small, evaluation results dipped below industry standard but were still satisfactory. As more event and tracking data is released by football data providers I will seek to build upon this foundational model by feeding more data, applying more features, and experimenting with other model (e.g. rndm forests + logistic regression).

## Sources
The following sources were used to create this project: <br>
Cavus, M. and Biecek, P., 2022, October. Explainable expected goal models for performance analysis in football analytics. In 2022 IEEE 9th International Conference on Data Science and Advanced Analytics (DSAA) (pp. 1-9). IEEE. <br>
Fernandez, J., Bornn, L. and Cervone, D. (2021) ‘A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions’, Machine Learning, 110, pp. 1389–1427. <br>
Pardo, P.M., 2020. Creating a model for expected goals in football using qualitative player information (Doctoral dissertation, Universitat Politècnica de Catalunya. Facultat d'Informàtica de Barcelona).
Singh, K. (2018) Introducing Expected Threat (xT). Available at: https://karun.in/blog/expected-threat.html (Accessed: 19 March 2024). <br>
Spearman, W., 2018, February. Beyond expected goals. In Proceedings of the 12th MIT sloan sports analytics conference (pp. 1-17). <br>
Sumpter, D. (2022) Expected goals including player positions, Expected Goals including player positions - Soccermatics documentation. Available at: https://soccermatics.readthedocs.io/en/latest/gallery/lesson7/plot_xG_tracking.html (Accessed: 19 March 2024). <br>
Umami, I., Gautama, D.H. and Hatta, H.R., 2021. implementing the Expected Goal (xG) model to predict scores in soccer matches. International Journal of Informatics and Information Systems, 4(1), pp.38-54. <br>
