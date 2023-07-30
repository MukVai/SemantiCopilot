# Positive Sentiments

result_df.query('Score == 1').sort_values('RoBERTa_pos', ascending = False)['Text'].values[0]
#check manually if the model gives apropriate results

result_df.query('Score == 1').sort_values('VADER_pos', ascending = False)['Text'].values[0]

result_df.query('Score == 1').sort_values('electra_pos', ascending = False)['Text'].values[0]


# Negative Sentiment

result_df.query('Score == 5').sort_values('RoBERTa_neg', ascending = False)['Text'].values[0]

result_df.query('Score == 5').sort_values('VADER_neg', ascending = False)['Text'].values[0]

result_df.query('Score == 5').sort_values('electra_neg', ascending = False)['Text'].values[0]

