# Sentiment Analysis

This project is an implementation of
[Sentiment Analysis of Microblog Data Streams](https://drive.google.com/drive/u/1/folders/1Mi0jOX-078q7qlJKZkrp6-y4eCBaFztZ) lab.

## Testing

The program prints the results of Topic Classification and 
Sentiment Classification respectively. It doesn't take 'irrelevant'
tweets into account, but filters it.

## New tweets classification

If you want to classify any new tweets which are not included
into testing dataset, you may do it following this way:
- Wait until testing dataset results are printed
- Print the number of new tweets to classify
- Print the tweets following the structure of tweets
 represented in datasets
 
 The classification results will be given following this structure:
 \<tweet text\>: \<topic\>, \<sentiment\>.