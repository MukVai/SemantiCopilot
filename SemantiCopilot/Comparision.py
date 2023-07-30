result_df.head()

sns.pairplot(data = result_df, vars = ['VADER_neg', 'VADER_neu', 'VADER_pos',
       'electra_neg', 'electra_neu', 'electra_pos', 'RoBERTa_neg',
       'RoBERTa_neu', 'RoBERTa_pos'], hue = 'Score', palette = 'tab10')
#generating a plot of ratings as given by the users (stars 1to 5)
plt.show()

