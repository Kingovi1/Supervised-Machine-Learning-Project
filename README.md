# INTRODUCTION

In the lending industry, one of the greatest challenges isn't the availability of funds, but rather the uncertainty surrounding borrowers' willingness to repay loans promptly and with the agreed-upon interest rates. This uncertainty has led to considerable issues within the lending sector, discouraging potential lenders from extending financial support. 

This project endeavors to address this challenge by harnessing the power of machine learning to analyze historical data on borrower behavior. By scrutinizing the actions of past loan recipients, we aim to develop a predictive model capable of assessing whether a prospective borrower is likely to fulfill their repayment obligations. 

Our dataset, sourced from LendingClub.com, provides invaluable insights into borrowers' characteristics and their repayment histories. By leveraging this data spanning from 2007 to 2010, we seek to create a model that can effectively predict whether a borrower will fully repay their loan. 

The columns in our dataset provide crucial information for this endeavor:
- **Credit Policy:** Indicates whether a customer meets the credit underwriting criteria of LendingClub.com.
- **Purpose:** Describes the intended use of the loan.
- **Interest Rate:** Reflects the rate at which interest is charged on the loan.
- **Installment:** Represents the monthly payment amount.
- **Annual Income:** The borrower's reported annual income.
- **Debt-to-Income Ratio:** Shows the proportion of debt relative to annual income.
- **FICO Credit Score:** Indicates the borrower's creditworthiness.
- **Credit Line Age:** The duration of the borrower's credit history.
- **Revolving Balance and Utilization:** Reflects the borrower's credit card balance and utilization rate.
- **Inquiries and Delinquencies:** Indicates recent inquiries by creditors and past delinquencies.
- **Public Records:** Reflects derogatory public records such as bankruptcy filings or tax liens.

By scrutinizing these factors, our goal is to empower lenders to make informed decisions, minimizing the risk of default and enabling the lending industry to operate more efficiently. Ultimately, this project aims to foster a more fluid lending ecosystem while reducing losses due to non-repayment.

# DATA PREPROCESSING

The dataset was found to be relatively clean, with no missing values. However, categorical features were present in the 'purpose' column. Since there weren't many unique values in this column, we encoded the categorical features by mapping them to integral values using a dictionary. This encoding simplifies the handling of categorical data during analysis and modeling.

 # EXPLORATORY DATA ANALYSIS
Matplotlib:

Histograms were plotted for the 'fico' column to visualize the distribution of FICO credit scores among borrowers. Similarly, histograms were created to compare the behavior of borrowers under the two possible credit policies, shedding light on their borrowing patterns based on different credit criteria.
Further insights into borrower behavior were gained by plotting histograms for the 'fico' column, segmented by loan repayment status. This comparison provided valuable insights into the FICO scores of borrowers who repaid their loans versus those who did not.

Seaborn:

A count plot was generated for the 'purpose' column to visualize the distribution of borrowing purposes among borrowers who repaid and those who did not. This analysis helped identify any trends or patterns in borrowing behavior based on different loan purposes.
A joint plot was created to explore the relationship between FICO scores and interest rates ('int.rate'). This visualization allowed us to assess whether there was any correlation between credit scores and the interest rates charged on loans.
Lastly, an lmplot was constructed to investigate the relationship between FICO scores and interest rates, colored by credit policy. This visualization enabled us to discern any differences in borrowing behavior between borrowers who repaid their loans and those who did not, based on credit policy and FICO scores.
By employing both Matplotlib and Seaborn for data visualization and exploration, we gained valuable insights into borrower behavior and identified potential patterns that may influence loan repayment outcomes. These insights will inform the development of predictive models to aid lenders in decision-making processes.

# MODELING APPROACH 
For modeling, we considered Decision Tree Classifier and RandomForestClassifier due to their effectiveness in classification tasks. These models were chosen as they can be fine-tuned for improved accuracy.

Evaluation Metrics:
We selected the classification report and confusion matrix to assess model performance.

Confusion Matrix:
It provides a comprehensive evaluation of classification model performance by comparing actual labels with predicted labels. Key components include:

- True Positive (TP): Correctly predicted positive class.
- True Negative (TN): Correctly predicted negative class.
- False Positive (FP): Incorrectly predicted positive class when the actual class is negative.
- False Negative (FN): Incorrectly predicted negative class when the actual class is positive.
- 
Classification Report:It is a summary of the performance of a classification model. It provides major evaluation metrics for each class in a classification problem.

Typically, a classification report includes the following metrics for each class:

Precision: The proportion of true positive predictions out of all positive predictions made by the model.
Recall: The proportion of true positive predictions out of all actual positive instances in the dataset.
F1-score: The harmonic mean of precision and recall, providing a balance between the two metrics.
Support: The number of actual occurrences of each class in the dataset.
Additionally, the classification report often includes an overall accuracy score for the entire model.

By studying the classification report, we can get insights into how well the model performs for each class and identify any imbalances or areas for improvement in the model's predictions.



# MODEL DEVELOPMENT

The train test split was imported  to split the data into training set and testing set  before fitting the training split of the data into the algorithm 

For DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,

                                             max_features=None, max_leaf_nodes=None, min_samples_leaf=1,

                                             min_samples_split=2, min_weight_fraction_leaf=0.0,

                                             presort=False, random_state=None, splitter='best')

The following hyper-parameter tuning  was used for the DecisionTreeClassifier

For RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                                                 max_depth=None, max_features='auto', max_leaf_nodes=None,

                                                 min_samples_leaf=1, min_samples_split=2,

                                                 min_weight_fraction_leaf=0.0, n_estimators=600, n_jobs=1,

                                                 oob_score=False, random_state=None, verbose=0,

                                                 warm_start=False)

The following hyper-parameter tuning  was used for the RandomForestClassifier



MODEL EVALUATION

For the DecisionTreeClassifier the classification report  brought out the following report  

        precision    recall  f1-score   support



          0       0.85      0.82      0.84      2431

          1       0.19      0.23      0.20       443



avg / total       0.75      0.73      0.74      2874

 And the confusion matrix brought 

[[1995  436]

 [ 343  100]]



For the RandomForestClassifier  the classification report

     precision    recall  f1-score   support



           0       0.84      0.99      0.91      2650

           1       0.36      0.02      0.03       511



    accuracy                                 0.84      3161

   macro avg       0.60      0.51    0.47      3161

weighted avg      0.76     0.84    0.77      3161



And for confusion matrix

[[2634   16]

 [ 502    9]]



For higher accuracy RandomForestClassifie was chosen for the model since it brought forward a greater accuracy



CONCLUSION

The RandomForestClassifier model achieved an overall accuracy of 84%, demonstrating its effectiveness in classifying the majority class. However, its performance varied across classes, with higher precision and recall for class 0(those that have not fully paid) compared to class 1(those that have paid).

Displaying SMLProject.py. 
