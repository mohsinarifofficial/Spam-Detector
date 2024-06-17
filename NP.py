# its a second catogry of extractiong useful information 
# it called text classification
# we use naive bayes theorem here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data=pd.read_csv("spam.csv",encoding="ISO-8859-1")
print(data.groupby("Category").describe())
def make_Spam(x):
    if x=="spam":
        return 1
    else:
        return 0
data["Spam"]=data.Category.apply(make_Spam)
X_Train,X_Test,Y_Train,Y_Test=train_test_split(data.Message,data.Spam,test_size=0.25)
v=CountVectorizer()
X_Train_Count=v.fit_transform(X_Train.values)
print(X_Train_Count)
X_Train_Count.toarray()[:3]
print(X_Train_Count)

#print(X_Train_Count)
model=MultinomialNB()
model.fit(X_Train_Count,Y_Train)
print(model.score(X_Train_Count,Y_Train))

email=["you won one million ","Looking forward to for with you sir"]
email_Count=v.transform(email)
p=model.predict(email_Count)
print(p)
for i in p:
    
    if p[i]==0:
        print(email[i]," is a HAM")
    else:
        print(email[i]," is a Spam")
