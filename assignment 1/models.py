from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

models = {
    "M1_LogisticRegression": LogisticRegression(max_iter=1000),
    "M2_DecisionTree": DecisionTreeClassifier(random_state=42),
    "M3_RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "M4_KNN": KNeighborsClassifier(),
    "M5_SVM": SVC()
}

print("Models initialized.")
