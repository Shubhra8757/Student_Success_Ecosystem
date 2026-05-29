# AI Student Success Ecosystem

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import accuracy_score

# reading dataset

df = pd.read_csv("college_student_placement_dataset.csv")

# checking first 20 rows

print(df.head(20))

# Shape of dataset

print("\nDataset Shape :", df.shape)

# removing unnecessary column

df.drop("College_ID", axis=1, inplace=True)

# converting text columns into numbers

df["Internship_Experience"] = df["Internship_Experience"].replace({
    "Yes": 1,
    "No": 0
})

df["Placement"] = df["Placement"].replace({
    "Yes": 1,
    "No": 0
})

# input and output

x = df.drop("Placement", axis=1)

y = df["Placement"]

# splitting dataset

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=1
)

# creating logistic regression model

model = LogisticRegression()

# training model

model.fit(x_train, y_train)

# testing model

predictions = model.predict(x_test)

# checking accuracy

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy :", accuracy * 100)

# confusion matrix

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix :")
print(cm)

# precision

precision = precision_score(y_test, predictions)

print("\nPrecision :", precision)

# recall

recall = recall_score(y_test, predictions)

print("Recall :", recall)

# sample student details

cgpa = 8.4
academic = 85
internship = 1
extra = 70
communication = 7
projects = 4
iq = 110
prev_sem = 8.2

# creating student input

student_data = [[
    iq,
    prev_sem,
    cgpa,
    academic,
    internship,
    extra,
    communication,
    projects
]]

# placement prediction

result = model.predict(student_data)

probability = model.predict_proba(student_data)

print("\nPlacement Probability :", probability[0][1] * 100)

# final result

if result[0] == 1:
    print("Student may get placed")

else:
    print("Student placement chances are low")

# student success score

score = (
    cgpa * 10 +
    academic * 0.4 +
    communication * 5 +
    projects * 5 +
    prev_sem * 5
)

print("\nStudent Success Score :", score)

# skill gap analysis

print("\nSkill Gap Analysis")

if communication < 5:
    print("- Improve communication skills")

if projects < 3:
    print("- Work on more projects")

if internship == 0:
    print("- Try getting internship experience")

if cgpa > 8:
    print("- Academic performance is very good")

# learning recommendation

print("\nRecommended Learning")

if communication < 5:
    print("- Communication Skills Course")

if projects < 3:
    print("- Build more Python and ML projects")

if cgpa < 7:
    print("- Focus on academics and study planning")

# career recommendation

print("\nSuggested Career Path")

if projects >= 4 and iq > 100:
    print("Data Science / Software Development")

elif communication > 7:
    print("Business Analyst")

else:
    print("Web Development")
