# LMSYS - Chatbot Arena Human Preference Predictions
By Nicole Michaud
5 August, 2024 

## Introduction

Models utilizing human feedback to improve predictions are quite popular right now. In fact, GPT-4, released earlier this year, is trained using a human feedback reward model.
In that instance, humans provided feedback for the responses they received from the model, and it took this feedback into account in order to improve its future responses.

The goal of this project is to use human feedback, not just to improve the performance of one chatbot model, but to predict which response from different models will be preferred by users. This could eventually be used to improve these models further, if we could uncover what kind of responses are more likely to be preferred by users.

In each row of the training data, two different chatbot models go head-to-head, and users have indicated which response to the prompt they prefer, either the response from model A ('response_a'), the response from model B ('response_b'), or if they preferred them both equally.

This information is represented as binary values (0 or 1) in the three target columns ('winner_model_a', 'winner_model_b', and 'winner_tie').

It should be noted that there are various biases that may impact which response users chose as the one they prefer.
These biases include positional bias, which may lead users to prefer the first response, verbosity bias may, which lead them to prefer the more verbose response, or self-enhancement bias, which may lead them to prefer the response that provides the most self-promotion.

I aim to create a model that takes the prompt and the two different responses as input and outputs the probability of each of the three outcomes: winner_model_a, winner_model_b, and winner_tie.


## Data Exploration

The training dataset provided contains a total of 57,477 'battles', or rows.


I was curious to see if there were certain chatbots that seem to win more often than others, regardless of whether they are the model a or model b in the battle.

I found that the top 5 chatbot models that most frequently win as model A are gpt-4-1106-preview, gpt-4-0613, gpt-3.5-turbo-0613, gpt-4-0314, and claude-2.1.

The top 5 chatbot models that most frequently win as model B are the same as the winners for model A, except instead of claude-2.1 in fifth place, we have claude-1.

In general, it would appear that the responses from the GPT chatbots are often preferred over the responses from other chatbots.


## Data Preparation

Some of the rows in the dataframe contain no information for one or more of the text features (they have a string length of 0). There are 137 rows with an empty 'prompt' feature, 147 rows with an empty 'response_a' feature, and 131 rows with an empty 'response_b' feature.

These empty features could cause problems in my modeling process, so I am going to remove rows that have any empty features. This leaves 57,119 rows in the dataset.

Next, because the input features for our model are all going to be text features, it is important to perform some cleaning and preprocessing to get these features ready to be tokenized and vectorized.

With the help of NLTK and regex, I use a function to make all letters lowercase, remove stopwords, stem the words, and remove any unnecessary characters or punctuation. 

By joining together the three text columns, I determined that the total number of unique words in the dataset is 444,900. This will be necessary for tokenization.

I remove any unnecessary features/columns that I won't need for modeling or making predictions.

<!-- I then train-test split the train dataset, so that I will have an unseen set of the data to evaluate my models against, before I generate predictions for the test data. -->

Both the trainset and testset features were tokenized, based on the total vocabulary length, vectorized, and then padded to all be of equal length.

In order to make sure the output of the model is a probability distribution of the three different outcomes (winner_model_a, winner_model_b, winner_tie), I am going to format this as a multi-class classification problem.

X_train is going to be the vectorized text of the three text features ('prompt', 'response_a', 'response_b'), and y_train is going to be an array of the three output features, which are already binary.


## Modeling


I created a Keras Sequential model as the baseline model, with an input layer of the correct shape of the inputs (the maximum sequence length of 2000 that I padded/truncated the text features to), an embedding layer for the text features, a couple LSTM layers,  multiple dense layers followed by LayerNormalization layers and Dropout layers to prevent overfitting, and a final output layer of three units for the three different possible outcomes with a Softmax activation function to indicate that these outcomes should mutually exclusive probabilities that add up to 1.

I instantiated the model and compiled it using the Adam optimizer and specified the loss as categorical cross-entropy, which is the same thing as log loss for multi-class classification.

Lastly, I fit the model to the training data, creating a validation split in this step.<!--  I utilized an early stopping callback to stop fitting early if the loss value isn't improving after two epochs. -->

<!-- I generated predictions for the testset I created, to see how they looked and if the values made sense to me (each row should be three positive numbers that add up to 1). -->

The baseline model had a validation loss of [1.0977]. We want the loss value to be as low as possible, so this value isn't bad, but it can probably be improved with hyperparameter tuning.

### Improving Model Performance

With Keras Tuner, I build a model that has the same types of layers as the baseline, but that will search for and use the best parameters for those layers. The hyperparameters I am choosing to use are the vector size of the embedding layer, the dropout rate for the Dropout layer, the units for the LSTM layer and the Dense layers, and the learning rate.

I then used the best hyperparameters, as determined by my tuner, to build a new model and fit it to the data for a total of 15 epochs (with a callback to stop early if the validation hasn't improved in 5 epochs).
The lowest (best) loss value for the validation set from this model was __. 
I saved this model's weights as my final model, then reloaded the model and used it to generate predictions for the test dataset.
I then saved these predictions to a CSV file to be submitted for the Kaggle competition.

## Conclusion

Using a Keras Sequential model and optimizing the model hyperparamters using Keras Tuner, I was able to create a model that can predict which chatbot response will be preferred by a user for a given prompt. 
The final model had a categorical cross-entropy (log loss) value of 1.0972.
This was an improvement from the baseline model's best loss value of 1.0989.


#### Next Steps:

- While this model was being evaluated on its log loss/categorical cross-entropy loss score, I also chose to print out the model's categorical accuracy values. I was able to improve the loss value of the model through hyperparameter tuning, however the accuracy scores remained fairly low (0.3520 at most). In the future, efforts should be made to improve these accuracy scores as well.
- I chose to build my own model from scratch for this task, however there is an option to instead use a pre-trained model and fine-tune it for the data being used. These pre-trained models typically perform very well. In the future, the perfomance of pre-trained models from Keras for this task could be compared to this model that I built to see which one has lower loss and/or higher accuracy.
- Another potential approach for improving the performance and efficiency of this model is to use FNet, which adds a Fourier Transform layer for "token mixing" [source], and has been shown to speed up training time and produce results that are comparable to transformer-based language models.
