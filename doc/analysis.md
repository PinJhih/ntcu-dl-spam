# enron_spam_data.csv
- columns: 'Message ID', 'Subject', 'Message', 'Spam/Ham', 'Date'
	- rename 'Spam/Ham' to 'label' in this project
- number of rows: 33716
	- spam: 17171
	- ham: 16545
- `Message` column
	- String length: quantile=[318, 688, 1529]
	- longest string: len=228353, Message ID=14254
	- Missing Data: 371 rows, 319 spam, 52 ham. pandas fills `NaN` for the missingg data by default. Replaced with string "missing" when preprocessing. 

- `Subject` column
	- String length: quantile=[20, 31, 45]
	- longest string: len=3153, Message ID=[5005, 22049]. The two rows have the same values of Subject, both miss the value of Message, and are both spam. 
	- missing data: 289 rows, all spam. pandas fills `NaN` for the missingg data by default. Replaced with string "missing" when preprocessing.


# email_classification.csv
- columns: 'email', 'label'
- number of rows: 179
	- spam: 79
	- ham: 100
- Content
	- longest string: 107

