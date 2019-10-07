# Classifying Credit Card Default
This project aims to see if credit card default can be predicted based on consumer payment/spending behavior. The sample 
dataset is from UCI's machine learning repository for a Taiwanese Bank. It includes 30,000 customers and their payment behavior
over a 6 month period. The payment behavior features can be grouped into 3 categories.
1. Timeliness, did the customer pay on time, late, or early and by how many months
2. Total amount, how much is on the customer's bill for the month
3. Amount paid, how much did the customer pay for that month's bill

## Engineered Features 
I created a new feature for 'perecent of bill paid' which took how much the customer paid for each month divided by that 
month's bill. I also created several dummy features for sex, education, marrital status, and age.

## Algorithms Used
* KNN
* Logistic Regression
* Decision Trees
* Random Forest
* Gradient Boosting Classifier
* Bernoulli Naive Bayes

## Results
Random Forest produced the best results with an AUC of 0.77 and recall score of 0.22. To improve the recall, I created a
confusion matrix and tuned the threshold to minimize those customers who were predicted as not in default when they actually
were in default. 
