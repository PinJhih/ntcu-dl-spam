# Projcet Descriptions
- First project of NLP: spam mail binary classifier.
- The dataset is typically unbalanced (normal > spam).

# Dataset
- [Email Classification (Ham-Spam)](https://www.kaggle.com/datasets/prishasawhney/email-classification-ham-spam), last update: Aril,2024
- [enron_spam_data](https://github.com/MWiechmann/enron_spam_data), last update: 2021
- [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), last update: 2016 (?)

# Considerations of Models
- simplicity
- performance
- computational resources

# Models to be tried
- LSTM
- BERT
	- Lightweight variants: DistilBERT, ALBERT
	- Advanced variants: RoBERTa (robustly optimized), DeBERTa
- RoBERTa
- XLNet

# Metrics
- For balanced dataset: accuracy
- For unbalanced dataset: precision, recall, f1, AUC, PR-AUC

# Experiment
- CV Methods: K-fold for balanced dataset, Stratified K-Fold for unbalanced dataset
