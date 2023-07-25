# Sentiment-Analysis
A Sentimental Analysis using 3 pre-trained Machine Learning Data Models.


In this project we will be working on multiple techniques that are:

1. VADER (Valence Aware Dictionary and Sentiment Reasoner) - Bag of words approach
2. Roberta Pre-trained Model from HuggingFace Pipeline
3. Electra Pre-trained Model from HuggingFace

-----------------------------------------------------------------------------------------------------------------------

# Valence Aware Dictionary and Sentiment Reasoner 
#### Uses a Bag-Of-Words approach Bag of Words Approach:

All the stop words are removed. (eg. and, the, of, etc.)
Each word is scored all the resultant is a combined score of the sentence.

Note: It is important to note that this mode does not account the relationship between words in a sentence. It just individually scores every word as either positive, negative or neutral. Then the whole is summed up to generate a cumulative sum score.
