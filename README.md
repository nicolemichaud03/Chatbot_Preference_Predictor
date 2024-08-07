# LMSYS - Chatbot Arena Human Preference Predictions
###### By Nicole Michaud
###### 5 August, 2024 
<a href="https://www.kaggle.com/code/nicolemichaud/lmsys-competition"> Link to notebook </a>

## Introduction

Models utilizing human feedback to improve predictions are quite popular right now. In fact, GPT-4, released earlier this year, is trained using a human feedback reward model.
In that instance, humans provided feedback for the responses they received from the model, and it took this feedback into account in order to improve its future responses.

The goal of this project is to use human feedback, not just to improve the performance of one chatbot model, but to <em>predict which response from different models will be preferred by users</em>. 
If we could uncover what kind of responses are more likely to be preferred by users, this could eventually be used to improve these models further.

In each row of the training data, two different chatbot models go head-to-head, and users have indicated which response to the prompt they prefer, either the response from model A ('response_a'), the response from model B ('response_b'), or if they preferred them both equally (they tied).


It should be noted that there are various biases that may impact which response users chose as the one they prefer:
- positional bias, which may lead users to prefer the first response
- verbosity bias may, which lead them to prefer the more verbose response
- self-enhancement bias, which may lead them to prefer the response that provides the most self-promotion

I aim to create a model that takes the prompt and the two different responses as input and outputs the probability of each of the three outcomes: winner_model_a, winner_model_b, and winner_tie.


## Data Exploration

The training dataset provided contains a total of 57,477 'battles', or rows.


I was curious to see if there were certain chatbots that seem to win more often than others, regardless of whether they are the model a or model b in the battle.

I found that the top 5 chatbot models that most frequently win as model A are gpt-4-1106-preview, gpt-4-0613, gpt-3.5-turbo-0613, gpt-4-0314, and claude-2.1.

The top 5 chatbot models that most frequently win as model B are the same as the winners for model A, except instead of claude-2.1 in fifth place, we have claude-1.

In general, it would appear that the responses from the GPT chatbots are often preferred over the responses from other chatbots.


## Data Preparation

The main features that will be used by the model to make predictions are all text features: "prompt", "response_a", and "response_b". 
Therefore, various natural language processing (NLP) techniques are employed to prepare the text features for modeling. 
These techniques include cleaning the text (removing stopwords, punctuation, numbers, and unnecessary characters, and making all letters lowercase), removing rows that contain empty strings, tokenizing the text, and vectorizing the text sequences to all be of the same length.
Tools such as NLTK (the Natural Language Toolkit from Scikit-learn) and Regex (regular expressions) were employed for these tasks.

The desired output of modeling is a probability distribution of the three different outcomes (winner_model_a, winner_model_b, winner_tie), so this is a <em>multi-class classification</em> problem.

X_train is the vectorized text of the three text features ('prompt', 'response_a', 'response_b'), and y_train is an array of the three output features, which are already binary.


## Modeling


The baseline model is a Keras Sequential model, with an input layer of the correct shape of the inputs (the maximum length the sequences were padded to), an embedding layer for the text features, a couple LSTM layers,  multiple dense layers followed by LayerNormalization layers and Dropout layers to prevent overfitting, and a final output layer of three units for the three different possible outcomes with a Softmax activation function.

The model was optimized by using the Adam optimizer and was evaluated based on categorical cross-entropy, which is the same thing as log loss for multi-class classification tasks.

In fitting the model, a validation split of 20% was implemented to compare model performance on unseen data.

This baseline model had a validation loss of [1.0989]. We want the loss value to be as low as possible, so this value isn't bad, but it can hopefully be improved with hyperparameter tuning.

### Improving Model Performance

A Keras Tuner model was built, which had the same types and number of layers as the baseline model, but that was set to search for the optimal parameter values to use for each layer. The optimal learning rate value was also searched for.

Using these optimal hyperparameters, a new model was compiled and fit to the training data, with a validation split. This model had a best validation loss value of 1.0972, which is an improvement from the baseline model.

This is the final model, and its weights were then saved and reloaded to be used to generate predictions for the test dataset.

These predictions were then saved to a CSV file.

## Conclusion

Using a Keras Sequential model and optimizing the model hyperparamters using Keras Tuner, I was able to create a model that can predict which chatbot response will be preferred by a user for a given prompt. 
The final model had a categorical cross-entropy (log loss) value of 1.0972.
This was an improvement from the baseline model's best loss value of 1.0989.


#### Next Steps:

- While this model was being evaluated on its log loss/categorical cross-entropy loss score, I also chose to print out the model's categorical accuracy values. I was able to improve the loss value of the model through hyperparameter tuning, however the accuracy scores remained fairly low (0.3520 at most). In the future, efforts should be made to improve these accuracy scores as well.
- I chose to build my own model from scratch for this task, however there is an option to instead use a pre-trained model and fine-tune it for the data being used. These pre-trained models typically perform very well. In the future, the perfomance of pre-trained models from Keras for this task could be compared to this model that I built to see which one has lower loss and/or higher accuracy.
- Another potential approach for improving the performance and efficiency of this model is to use FNet, which adds a Fourier Transform layer for "token mixing" [source], and has been shown to speed up training time and produce results that are comparable to transformer-based language models.
- Future research/modeling could take the various possible biases into account and incorporate features that mitigate these biases