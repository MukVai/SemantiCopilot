# Importing necessary transformers and softmax to be used on the created model.
from transformers import AutoTokenizer # Automatically Tokenises the Review - Similar to what we did using NLTK.
from transformers import AutoModelForSequenceClassification # A kind of model that the hugging face has 
from scipy.special import softmax # It will smoothen the output between 0 and 1.

# For removing the warnings related to tenserflow plugins.
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, 
#                       message="unable to load libtensorflow_io_plugins.so*")
# warnings.filterwarnings("ignore", category=UserWarning, 
#                       message="file system plugins are not loaded*")

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# tokenizer and model will put down the model weights that helps doing transfer learning from twitter reviews to
# Amazon Fine Food Reviews

# After running these we have a pre-trained model imported and ready and the tokenizer which we can apply on the text.

# Results of Vader on an example review.
print(eg)
sia.polarity_scores(eg)

# Running same example for RoBERTa Model
# 1st encoding the review in the forms of 0s and 1s, so that RoBERTa model is able to understand it.
#tokenizer(eg, return_tensors = 'pt') # pt stands for py-torch.
encoded_text = tokenizer(eg, return_tensors = 'pt') # pt stands for py-torch.
# model(**encoded_text)
output = model(**encoded_text) # implementing the RoBERTa model on the eg text.
scores = output[0][0].detach().numpy() # converting the output from tensorflow to numpy so that can store locally.
scores = softmax(scores) # making the score softer between 0 and 1.
# defining scores dictionary to store these values.
scores_dict = {
                'RoBERTa_neg' : scores[0],
                'RoBERTa_neu' : scores[1],
                'RoBERTa_pos' : scores[2]
              }
print(scores_dict)

# Creating a function out of the above so that we can implement the same for whole dataset.
def polarity_scores_RoBERTa(eg):
    # Running same example for RoBERTa Model
    # 1st encoding the review in the forms of 0s and 1s, so that RoBERTa model is able to understand it.
    #tokenizer(eg, return_tensors = 'pt') # pt stands for py-torch.
    encoded_text = tokenizer(eg, return_tensors = 'pt') # pt stands for py-torch.
    # model(**encoded_text)
    output = model(**encoded_text) # implementing the RoBERTa model on the eg text.
    scores = output[0][0].detach().numpy() # converting the output from tensorflow to numpy so that can store locally.
    scores = softmax(scores) # making the score softer between 0 and 1.
    # defining scores dictionary to store these values.
    scores_dict = {
                    'RoBERTa_neg' : scores[0],
                    'RoBERTa_neu' : scores[1],
                    'RoBERTa_pos' : scores[2]
                  }
    return scores_dict

# Try running individually.
VADER_result_rename
RoBERTa_result

