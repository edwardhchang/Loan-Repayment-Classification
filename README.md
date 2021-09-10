# Overview

Peer-to-peer(P2P) lending is a growing space that is overtaking an area generally inhabited by traditional financial institutions. P2P lending resembles crowd-funding but for the loan space, where investors select which loans they will partially fund based upon borrower characteristics and loan details.  Loans provide the opportunity to link up lenders with those in need of capital. The investors make profit due to interest tacked onto the loan. Borrowers benefit as they can find competitive or even cheaper interest rates for loans in an efficient and cost-effective way (online). PricewaterhouseCoopers LLP projects P2P lending originations to grow to 150 billion USD by 2025. With a cost, the P2P lending industry is inherently risky due to the lack of collateral in loan agreements as well as information assymetry between the investor and borrower. Sadly, not all loans see their principal paid until the end, and when the loan is not paid-in-full (PIF) then the loan is charged-off. When this occurs, the lender loses money and due to the chance of charge-off when it comes to microloans, it is imperative that the platforms providing this link between lenders and borrowers have a risk model to mitigate potential charge-off based on borrower information. **This study aims to train several machine learning models to classify if a borrower will repay the loan or potentially charge-off using Lending Tree data from 2007 - 2018.** Lending Tree has been providing open-source data from their loan applications, but have since halted in 2020. The guiding metric will be the area under the ROC curve (AUC), specifically the precision-recall AUC score due to the severity of false negatives, or loans that our model incorrectly predicts as valid, but actually will most likely charge-off. We will also keep an eye on the recall score of our predictions as a higher recall score indiciates less false negatives. 

# Executive Summary
Our data was imported as a .csv file from Kaggle with 2.26 million samples with 151 features. We removed features that would not be included on a traditional loan screening as well as features alluding to joint applications as we only dove into individual applications. We also filtered our all samples that did not have a loan status of 'Fully Paid' or 'Charged-Off'. We simplified various categorical variables, specifically, 'term', 'emp_length',and 'home_ownership'. FICO score came in ranges and so we had to feature engineer the average of the FICO score as well as drop outliers created due to the averaging. We also engineered a feature using the datetime values 'issue_d', which indicates the issue date of the loan, as well as the 'earliest_cr_line', which indicates the year and month the earliest credit file was created. We imputed null values for columns, using median, mode, and mean depending on the data type. We lastly transformed the annual income feature to its log form due to a heavy right skew found in exploratory data analysis. 

We chose to model using logistic regressions, random forest classifiers, K-nearest neighbors classifiers, and deep neural networks. We selected these models as they represented different types of machine learning algorithms: linear, non-linear, ensemble, and articial neural network. For each model, we ran a default model and then tuned for hyperparameters using cross-validated grid-search. We ran the Random Forest model first with the intention of using it for feature selection. 

For the random forest classifier, we tuned the number of trees ('n_estimators'), the max number of features, the max depth of each tree, and whether to bootstrap. For the logistic regression, we tuned the the C-parameter which is the inverse of lambda, or the regularization strength. We also tuned for 'class_weight' to see how the model would deal with imbalanced classes on its own, as well as tuning for max iterations as the model failed to converage occasionally. For the K-Nearest Neighbors classifier, we tuned the number of neighbors, the distance metric used for the tree, as well as the weights used for function. Our deep neural networks required manual tweaking of network structure and regularization due to training times using grid search. We finally used a grid search using dropout for regularization as well as tuning for number of neurons in each layer and number of hidden layers. We created predictions and probabilities for each model and exported each for evaluation. 
Overall, logistic regressions had the fastest training times, whereas K-Nearest Neighbors took the longest. We increased the batch size when fitting our neural networks to speed up each epoch, but the networks would, without a doubt, show higher scores if run on a larger number of epochs. Our baseline model was created using a DummyClassifer from the scikit-learn machine learning library. 

# Directory

- **data** folder
  - **accepted_2007_to_2018Q4.csv** : Lending Club raw loan data from Kaggle
  - **deep_probs.csv** : Prediction probabilities from Deep Neural Network
  - **knn_probs.csv**  : Prediction probabilities from K-Nearest Neighbors model
  - **loan_all.csv**   : Dataset with all features after initial data cleaning
  - **loan_dropped.csv** : Dataset with all null values dropped
  - **loan_rf_features.csv** : Dataset with features using Random Forest for feature selection
  - **lr_probs.csv** : Prediction probabilities for Logistic Regression model
  - **rf_probs.csv** : Prediction probabilities for Random Forest model
- **data_dictionary** folder
  - **LCDataDictionary.xlsx** : Lending Club Data Dictionary
- **images** folder
  - **fico_boxplot.png** : FICO Score boxplot
  - **loan_over_time.png** : Loan applications over time plot
  - **loan_purpose.png** : Loans broken down by purpose barchart
  - **loan_status.png** : Loans broken down by loan status
  - **pairplot.png** : Pairplot of all features
  - **pr_curve.png** : Precision-Recall Curve of all models
  - **rf_confusion_matrix.png** : Confusion Matrix for Random Forest model
  - **rf_important_features.png** : Feature Importance from Random Forest model barchart
- **presentation** folder
  - **Peer-to-Peer Loan Default Prediction.pptx** : PowerPoint presentation of findings
- **code** folder: 
  - **Data_Cleaning_and_EDA.ipynb** : Data collection, cleaning, and exploratory data analysis
  - **Deep_Learning_model.ipynb** : Neural Network Modeling
  - **KNN_Model.ipynb** : K-Nearest Neighbors Modeling
  - **Log_Reg_Model.ipynb** : Logistic Regression Modeling
  - **Random_Forest_model.ipynb** : Random Forest Modeling
- **Evaluation_and_Conclusions.ipynb** : Evaluation and Conclusion
- **Executive Summary.ipynb** : Executive Summary
- **README.md** : Readme file
 
# Data Dictionary
 - Please refer to 'LCDataDictionary.xlsx' in the **data_dictionary** folder
 
# Necessary Files/Libraries

- Tensorflow / Keras library necessary to run code
- Scikit-Learn necessary to run code - ships with recent version of Anaconda

# Conclusions and Further Improvements
### Model Evaluation
| Model               | Accuracy Score | Precision Score | Recall Score | AUC-ROC Score | Precision-Recall AUC Score |
|---------------------|----------------|-----------------|--------------|---------------|----------------------------|
| Logistic Regression | 0.9015         | 0.7912          | 0.6853       | 0.9458        | 0.7968                     |
| Random Forest       | 0.9067         | 0.7613          | 0.7725       | 0.9503        | 0.8249                     |
| K-Nearest Neighbors | 0.8828         | 0.7861          | 0.5635       | 0.9151        | 0.7567                     |
| Deep Neural Net     | 0.9064         | 0.7596          | 0.7738       | 0.9506        | 0.8233                     |


Our Random Forest classifier performed the best when observing the Precision-Recall AUC score and with our Random Forest model, we can also extract feature importance. We found that FICO scores, interest rate, debt-to-income-ratio, installment size, and average current balance were important in our models. We also examined the coefficients from our logistic regression model since it still performed well, although with a much worse recall score. In terms of accuracy, it still ranked well  with a 90.1% accuracy score, so we may derive some explanatory power. Increases in FICO score led to a huge decrease in likelihood of default (94%), while annual income also saw the same relationship albeit at 12%. Increases in debt-to-income ratio, installment, and term saw increases in likelihood of default.

With this being a work in progress, we hope to improve our model by implementing several strategies with our data, modeling, and further studies. Due to the case that we have imbalanced classes, we will implement random undersampling as well as SMOTE with our minority class. Our data set is fairly large so it does not seem out of this world to undersample our majority class as we had ~1 million samples of this specified class. We would also look to implement probability tuning and threshold tuning to extract strong effectiveness out of our models in terms of our evaluation metrics (recall and precision-recall area-under-the-curve score). In addition to this, we were only able to model using 4 different algorithms, but seeing how our ensemble model (Random Forest) performed, we will also look to model using XGBoost and Extra Trees. Support Vector Machine classifiers also perform well due to the usage of kernel tricks but generally perform better on sparse data. There have been strategies created to aid in SVMs modeled on large data sets, specifically the minimum enclosing ball clustering method. We can also improve our models by reducing the complexity of our data such as removing unnecessary categorical features, as many of them were insignficant in the predictions of our models. We can also reduce the number of correlated features and deploy Recursive Feature Elimination to aid in our linear algorithms. 


# Sources

 1. 'Peer-to-peer loan acceptance and default prediction with artificial intelligence'. Turiel, Aste 2020 June. Royal Society Open Science <https://royalsocietypublishing.org/doi/10.1098/rsos.191649#d1e2778>
 2. 'Loan Repayment Prediction Using Machine Learning Algorithms'. Hang, Chang 2019. UCLA - eScholarhip. <https://escholarship.org/uc/item/9cc4t85b#main>
 3. 'A study on predicting loan default based on the random forest algorithm'. Zhu, Qiu, et. al 2019. Procedia Computer Science. <https://www.sciencedirect.com/science/article/pii/S1877050919320277>
 4. 'How to Use ROC Curves and Precision-Recall Curves for Classification in Python' Brownlee 2018. Machine Learning Mastery. <https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/>
 5. 'Financial Innovation and Borrowers: Evidence from Peer-to-Peer Lending' Balyuk 2019. Federal Deposit Insurance Corporation (FDIC). <https://www.fdic.gov/bank/analytical/fintech/papers/balyuk-paper.pdf>
 6. 'Determinants of Loan Performance in P2P Lending'. Möllenkamp 2017. University of Twente. <http://essay.utwente.nl/72876/1/M%C3%B6llenkamp_BA_BMS.pdf>
 7. 'A Survey of Predictive Modelling under Imbalanced Distributions'. Branco, Torgo, Ribeiro 2015. Cornell University. <https://arxiv.org/abs/1505.01658>
 8. 'Asymmetric Information, Bank Lending and Implicit Contracts: A Stylized Model of Customer Relationships'. Sharpe 1990. The Journal of Finance Vol. 45, No. 4. <https://www.jstor.org/stable/2328715>