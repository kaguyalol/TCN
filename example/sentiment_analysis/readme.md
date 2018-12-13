we use the model tcn to handel the sentiment analysis task. 
To deal with variate length variables, we add a rnn layer after the tcn block. 
Since the out put of the tcn block is a sequence of the same shape as the input, 
we use lstm to analysis the sequence and output the last hidden state as the feature used to do the classification. 
The class is inbalanced in the dataset. 
Thus we use auc of roc as a evalution metrics. 
We can reach to a acu of 0.82 on the test dataset after 3000 steps. 

