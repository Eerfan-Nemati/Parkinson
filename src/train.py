import pandas as pd
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score , recall_score ,precision_score , confusion_matrix ,f1_score
from imblearn.pipeline import Pipeline
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
# Import required libraries
data = pd.read_csv("data/Parkinsson disease.csv")

#loading the data from file csv in the folder by pandas
print(data.shape)
print(data.head())

#showing data shape  and printing first 5 rows of the dataframe
print(data.info())

#Data Exploration
print(data.describe())


data = data.drop(["name"] , axis=1)

# Remove the 'name' column before creating the correlation heatmap
data.isnull().sum()

#checking for  missing values in dataframe
print(data["status"].value_counts())

# Check the class distribution in the 'status' column
data["status"].value_counts().plot(
    kind ="bar" ,
    figsize=(6,6) ,
    rot = 360 ,
    color = [(0.5,1,1)]
)
plt.savefig("images/status histogram.png" , dpi=300, bbox_inches="tight")

# Plot a histogram of the target variable (status)
plt.figure(figsize=(20,18))
corr = data.corr()
sns.heatmap(corr ,
          cmap="RdBu" ,
          annot=True  ,
          square=True,
          fmt=".2%"  )
plt.title("Correlation Heatmap")
plt.savefig("images/heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# Calculate the correlation matrix and display it as a heatmap
plt.figure(figsize=(20,20))
for number , column in enumerate(data.columns , 1)  :
    plt.subplot(4,6,number)
    sns.boxplot(data[column].dropna() , color="green")
    plt.title(f"Box Plot for {column}")
plt.savefig("images/boxplot", dpi = 300,  bbox_inches="tight")
plt.show()

# Create box plots for all numerical features to detect outliers

x = data.drop(["status"], axis = 1)
y = data["status"]
print(f"x.shape : {x.shape}  y.shape : {y.shape}")

#splitting to data to x y
x_train , x_test , y_train , y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Check the class distribution in the 'status' column
pipline_knn = Pipeline([
    ("randomover" , RandomOverSampler(random_state=42)),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())

])

pipline_svc = Pipeline([
    ("randomover" , RandomOverSampler(random_state=42)),
    ("scaler", StandardScaler()),
    ("svc", SVC( ))

])

pipline_nb = Pipeline([
    ("randomover" , RandomOverSampler(random_state=42)),
    ("scaler", StandardScaler()),
    ("nb", GaussianNB())

])


# Create machine learning pipelines
param_knn = {
    "knn__n_neighbors": [2,3,4,5, 7,9,10],

}
grid_knn = GridSearchCV(pipline_knn, param_knn , cv = 5)
grid_knn.fit(x_train, y_train)

param_svc = {
    "svc__kernel" : [ "rbf" ,"linear" ],
    "svc__C": [0.1,1,10]

}
grid_svc = GridSearchCV(pipline_svc, param_svc , cv = 5)
grid_svc.fit(x_train, y_train)
pipline_nb.fit(x_train, y_train)

print(grid_knn.best_params_)
print(grid_knn.best_score_)
print()
print(grid_svc.best_params_)
print(grid_svc.best_score_)

# Perform hyperparameter tuning using GridSearchCV
cross_knn = cross_val_score(
    pipline_knn,
    x_train,
    y_train,
    cv = 5
)

cross_svc = cross_val_score(
     pipline_svc,
    x_train,
    y_train,
    cv = 5
)

cross_nb = cross_val_score(
    pipline_nb,
    x_train,
    y_train,
    cv = 5)

print(f"""
cross_knn : {cross_knn}
knn_mean : {cross_knn.mean():.3f} ± knn_std : {cross_knn.std()*100:.2f} %

cross_svc : {cross_svc}
svc_mean : {cross_svc.mean():.3f} ± svc_std  : {cross_svc.std()*100:.2f} %

cross_nb : {cross_nb}
nb_mean : {cross_nb.mean():.3f} ± nb_std  : {cross_nb.std()*100:.2f} %
""")

# Evaluate models using Cross-Validation
best_svc = grid_svc.best_estimator_

best_knn = grid_knn.best_estimator_

y_pred_knn = best_knn.predict(x_test)
x_train_knn = best_knn.predict(x_train)

acc_knn = accuracy_score(y_test , y_pred_knn)
acc_knn_train = accuracy_score(x_train_knn , y_train)


gap = acc_knn_train - acc_knn

if gap <= 0.1:
    print("No significant overfitting.")
else:
    print(f"Possible overfitting (gap = {gap:.3f})")

recall_knn = recall_score(y_test , y_pred_knn)
precision_score_knn = precision_score(y_test , y_pred_knn)
f1_knn = f1_score(y_test , y_pred_knn)
confusion_matrix_knn = confusion_matrix(y_test,y_pred_knn)


# Evaluate model performance and check for overfitting
print(f"Accuracy in knn : {acc_knn:.3f} \n Precision in knn : {precision_score_knn:.3f}  \n Recall in knn :  {recall_knn:.3f} \n f1 in knn : {f1_knn:.3f}  \n Confusion matrix in knn : \n {confusion_matrix_knn} ")
y_pred_svc = best_svc.predict(x_test)
x_train_svc = best_svc.predict(x_train)


acc_svc = accuracy_score( y_test , y_pred_svc)
acc_svc_train = accuracy_score(x_train_svc , y_train)

gap_svc =acc_svc_train - acc_svc
if gap_svc <= 0.1:
    print("No significant overfitting.")
else :
    print(f"Possible overfitting (gap = {gap_svc:.3f})")


precision_score_svc = precision_score( y_test , y_pred_svc)
recall_svc = recall_score( y_test , y_pred_svc )
f1_svc = f1_score( y_test , y_pred_svc )

confusion_matrix_svc = confusion_matrix(y_test,y_pred_svc)

# Evaluate model performance and check for overfitting
print(f"Accuracy in svc : {acc_svc:.3f} \n Precision in svc : {precision_score_svc:.3f}  \n Recall in svc :  {recall_svc:.3f} \n f1 in svc {f1_svc:.3f} \n Confusion matrix in svc : \n {confusion_matrix_svc} ")

y_pred_nb = pipline_nb.predict(x_test)
x_train_nb = pipline_nb.predict(x_train)


acc_nb = accuracy_score( y_test,y_pred_nb)
acc_nb_train = accuracy_score(x_train_nb , y_train)

gap_nb = acc_nb - acc_nb_train
if gap_nb <= 0.1:
    print("No significant overfitting.")
else:
    print(f"Possible overfitting (gap = {gap_nb:.3f})")



precision_score_nb = precision_score( y_test , y_pred_nb)

recall_nb = recall_score(y_test , y_pred_nb)
f1_nb = f1_score( y_test , y_pred_nb )
confusion_matrix_nb = confusion_matrix(y_test,y_pred_nb)

# Evaluate model performance and check for overfitting
print(f"Accuracy in Naive_Bayes : {acc_nb:.3f} \n Precision in Naive_Bayes : {precision_score_nb:.3f}  \n Recall in Naive_Bayes :  {recall_nb:.3f} \n f1 in nb : {f1_nb :.3f} \n Confusion matrix in Naive_Bayes :\n {confusion_matrix_nb} ")
models = pd.DataFrame({
    "Model": ["Naive_Bayes", "KNN", "SVC" ],
    "Accuracy": [acc_nb, acc_knn, acc_svc],
    "Precision": [precision_score_nb, precision_score_knn, precision_score_svc],
    "Recall": [recall_nb, recall_knn, recall_svc],
    "F1": [f1_nb, f1_knn, f1_svc]

})
result = models.sort_values( by="Model", ascending=False )
result =  result.round (3)
print(result)
result.plot(kind="barh" )
plt.title("Final_metrix")
plt.savefig("images/final_metrix.png")
plt.show()

# Compare model performance
joblib.dump(best_knn, "parkinsons_classification.joblib")
joblib.dump(best_svc, "parkinsons_classification.joblib")
joblib.dump(pipline_nb, "parkinsons_classification.joblib")

# Load the saved model

