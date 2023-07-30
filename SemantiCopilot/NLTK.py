# Basics of NLTK and Data Reading
# Importing the necessary Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt # for plotting

# Setting style sheet that we will be using for plots
plt.style.use('ggplot')

import nltk

# Reading the data from uploaded databse on kaggle notebook
df = pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')

# Run to see the contents of the CSV file.
df.head()
df['Text'].values[0] # We will be running sentimental on this row of data in our entire dataset.

# Checking and Considering size of the data - Reducing it Down
print(df.shape) # Print the size of the dataset
df = df.head(10000) # To reduce the time only considering 500 reviews as of now.
print(df.shape) # Will only show 500 rows now.

df.head() # for the sake of reference whenever theres need to check the data.

# QUICK DATA ANALYSIS (EDA)
# Checking the score counts and then sorting them in ascending order and plotting them as a bar graph.
# Try running individually.
df.score.value_count()
df.['Score'].value_count.sort_index()

ch1 = df['Score'].value_counts().sort_index().plot(kind = 'bar', title = 'Count of number of Scores(Stars out of 5)', figsize = (10,5))

ch1.set_xlabel('Star Rating out of 5')
ch1.set_ylabel('Numeric Count')
plt.show() # To view plot

# Cleaning the Data Set using Basic NLTK
# Let's consider an example

eg = df['Text'][100] # Can take any value - Taking 100th review text here.
print(eg) # Printing the 100th review.

# Tokenizing
tokens = nltk.word_tokenize(eg) # Breaking them in chunks of words and assinging it to a variable called tokens.
tokens[:10] # Showing only 1st 10 tokens.

# Tagging
tagged = nltk.pos_tag(tokens) # Assigning each token its part of speech
tagged[:10] # Showing only 1st 10 tags

# Chunking
# Putting tokens along with their tags in form of chunks to make them simpler to work upon as a whole.
entities = nltk.chunk.ne_chunk(tagged[:10]) # will have to store in a variable otherwise will give error.
entities.pprint() #pprint stands for pretty print, there is no attribute 'print' for this object.
