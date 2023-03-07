# credit-model-performance

## Description
Calculates the Gini coefficient and C-stat metrics for credit model that predicts the probability of default. 

## Assumptions
This monitor assumes that the data (test or production data) includes: 
- a column that provides a probability of default for each prediction from the model
- a column that provides the actual for the given record (i.e. whether the loan was defaulted)

## Inputs
Required Items in the Schema:
- Score Column: identifies the column that contains the scores/predictions in the data set
- Label Column: identifies the column that contains the actual results in the data set
