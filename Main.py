# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

# %%
dataset = pd.read_csv('spam_Emails_data.csv')
dataset.head()

# %%
dataset.isnull().sum()

# %%
dataset=dataset.fillna("NaN")

# %%
dataset.isnull().sum()

# %%
x = dataset.iloc[:,-1].values 
y = dataset.iloc[:,:-1].values

# %%
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
tf = TfidfVectorizer(stop_words='english')

# %%
x_train = tf.fit_transform(x_train)
x_test = tf.transform(x_test)

# %%
spam_detector_logistic = LogisticRegression(solver='liblinear',random_state=1)
spam_detector_navies = MultinomialNB()

# %%
spam_detector_navies.fit(x_train,y_train)
y_pred = spam_detector_navies.predict(x_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# %%
spam_detector_logistic.fit(x_train,y_train)
y_pred2 = spam_detector_logistic.predict(x_test)

print("Classification Report:")
print(classification_report(y_test, y_pred2))

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred2))


