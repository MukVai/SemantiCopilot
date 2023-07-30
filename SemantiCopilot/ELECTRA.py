from transformers import AutoTokenizer, ElectraForSequenceClassification
from scipy.special import softmax
import pandas as pd
from tqdm import tqdm
import logging

MODELE = "google/electra-base-discriminator"
tokenizerE = AutoTokenizer.from_pretrained(MODELE)
# Suppress the warning message
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

modelE = ElectraForSequenceClassification.from_pretrained(MODELE, num_labels=3)

def polarity_scores_electra(text):
    encoded_textE = tokenizerE(text, return_tensors='pt', truncation=True, padding=True)
    outputE = modelE(**encoded_textE)
    scoresE = outputE.logits.detach().numpy()[0]
    scoresE = softmax(scoresE)
    scores_dictE = {
        'electra_neg': scoresE[0],
        'electra_neu': scoresE[1],
        'electra_pos': scoresE[2]
    }
    return scores_dictE

# Run the polarity score on the selected dataset
res = {} # a dict to maintain polarity data with row id 
for i, row in tqdm(df.iterrows(), total = len(df)): #iterating over the dataframe with row id and row data and generating a progress bar, keeping the length of the data frame as length of the progress bar
    try:
        text = row['Text']
        id = row['Id']
        VADER_result = sia.polarity_scores(text) # Calculating result according to VADER Model.
        VADER_result_rename = {} # defining array to store renamed VADER result.
        for key, value in VADER_result.items(): # renaming result so that it can be understood that its particularly VADER's result.
            VADER_result_rename[f"VADER_{key}"] = value
        RoBERTa_result = polarity_scores_RoBERTa(text) # Calculating result according to VADER Model.
        electra_result = polarity_scores_electra(text)
        all_result = {**VADER_result_rename, **electra_result, **RoBERTa_result} # Combining both the dictionaries to print as a single result.
        # have to store this combined result into a new dictionary
        res[id] = all_result
    except RuntimeError:
        print(f'Broke for ID : {id}')

  #implementing the certaing transformers over the obtained results.
result_df = pd.DataFrame(res).T # a better visualization of the data stored in res
result_df = result_df.reset_index().rename(columns = {'index' : 'Id'}) # df stands for dataframe
result_df = result_df.merge(df, how = 'left') # Merging the results with the original dataset.

result_df.head()

