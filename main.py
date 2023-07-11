
import pandas as PD
import numpy as py
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

FileName = ['2_Desktop_Keyboard.csv', '3_Desktop_Keyboard.csv', '4_Desktop_Keyboard.csv',
            '5_Desktop_Keyboard.csv', '6_Desktop_Keyboard.csv', '7_Desktop_Keyboard.csv', '8_Desktop_Keyboard.csv',
            '9_Desktop_Keyboard.csv', '10_Desktop_Keyboard.csv', '11_Desktop_Keyboard.csv', '12_Desktop_Keyboard.csv',
            '13_Desktop_Keyboard.csv', '14_Desktop_Keyboard.csv', '15_Desktop_Keyboard.csv', '16_Desktop_Keyboard.csv',
            '18_Desktop_Keyboard.csv', '19_Desktop_Keyboard.csv', '20_Desktop_Keyboard.csv',
            '21_Desktop_Keyboard.csv', '22_Desktop_Keyboard.csv', '23_Desktop_Keyboard.csv', '24_Desktop_Keyboard.csv',
            '25_Desktop_Keyboard.csv', '26_Desktop_Keyboard.csv', '27_Desktop_Keyboard.csv', '28_Desktop_Keyboard.csv',
            '29_Desktop_Keyboard.csv', '30_Desktop_Keyboard.csv', '31_Desktop_Keyboard.csv', '32_Desktop_Keyboard.csv',
            '33_Desktop_Keyboard.csv', '34_Desktop_Keyboard.csv', '35_Desktop_Keyboard.csv', '36_Desktop_Keyboard.csv',
            '37_Desktop_Keyboard.csv', '38_Desktop_Keyboard.csv', '39_Desktop_Keyboard.csv', '40_Desktop_Keyboard.csv',
            '41_Desktop_Keyboard.csv', '42_Desktop_Keyboard.csv', '43_Desktop_Keyboard.csv', '44_Desktop_Keyboard.csv',
            '45_Desktop_Keyboard.csv', '46_Desktop_Keyboard.csv', '47_Desktop_Keyboard.csv', '48_Desktop_Keyboard.csv',
            '49_Desktop_Keyboard.csv', '50_Desktop_Keyboard.csv', '51_Desktop_Keyboard.csv', '52_Desktop_Keyboard.csv',
            '53_Desktop_Keyboard.csv', '54_Desktop_Keyboard.csv', '55_Desktop_Keyboard.csv', '56_Desktop_Keyboard.csv',
            '57_Desktop_Keyboard.csv','58_Desktop_Keyboard.csv', '59_Desktop_Keyboard.csv', '60_Desktop_Keyboard.csv',
            '61_Desktop_Keyboard.csv', '62_Desktop_Keyboard.csv', '63_Desktop_Keyboard.csv', '64_Desktop_Keyboard.csv',
            '65_Desktop_Keyboard.csv', '66_Desktop_Keyboard.csv', '67_Desktop_Keyboard.csv', '68_Desktop_Keyboard.csv',
            '69_Desktop_Keyboard.csv', '70_Desktop_Keyboard.csv', '71_Desktop_Keyboard.csv', '72_Desktop_Keyboard.csv',
            '73_Desktop_Keyboard.csv', '74_Desktop_Keyboard.csv', '75_Desktop_Keyboard.csv', '76_Desktop_Keyboard.csv',
            '77_Desktop_Keyboard.csv', '78_Desktop_Keyboard.csv', '79_Desktop_Keyboard.csv', '80_Desktop_Keyboard.csv',
            '81_Desktop_Keyboard.csv', '82_Desktop_Keyboard.csv', '83_Desktop_Keyboard.csv', '84_Desktop_Keyboard.csv',
            '85_Desktop_Keyboard.csv', '86_Desktop_Keyboard.csv', '87_Desktop_Keyboard.csv', '88_Desktop_Keyboard.csv',
            '89_Desktop_Keyboard.csv', '90_Desktop_Keyboard.csv', '91_Desktop_Keyboard.csv', '92_Desktop_Keyboard.csv',
            '93_Desktop_Keyboard.csv', '94_Desktop_Keyboard.csv', '95_Desktop_Keyboard.csv', '96_Desktop_Keyboard.csv',
            '97_Desktop_Keyboard.csv', '98_Desktop_Keyboard.csv', '99_Desktop_Keyboard.csv', '100_Desktop_Keyboard.csv',
            '101_Desktop_Keyboard.csv', '102_Desktop_Keyboard.csv', '103_Desktop_Keyboard.csv', '104_Desktop_Keyboard.csv',
            '105_Desktop_Keyboard.csv', '106_Desktop_Keyboard.csv', '107_Desktop_Keyboard.csv', '108_Desktop_Keyboard.csv',
            '109_Desktop_Keyboard.csv', '110_Desktop_Keyboard.csv', '111_Desktop_Keyboard.csv', '112_Desktop_Keyboard.csv',
            '113_Desktop_Keyboard.csv', '114_Desktop_Keyboard.csv', '115_Desktop_Keyboard.csv', '116_Desktop_Keyboard.csv',
            '117_Desktop_Keyboard.csv']

demographic_data = PD.read_csv("sample.csv")

df_cols = ['Key', 'User_ID', 'Dept', 'Age', 'Key_Mean', 'Key_STD']
df_cols2 = ['Digraph', 'K_PP_Time', 'K_RR_Time', 'K_PR_Time', 'K_RP_Time', 'K_PP_STD', 'K_RR_STD', 'K_PR_STD', 'K_RP_STD', 'Typing_Speed', 'Error_Frequency', 'L1_Gender']
df_cols3 = ['X-axis_Acc', 'Y-axis_Acc', 'Absolute_Acc', 'Pressure']

OutputDF = PD.DataFrame(columns=df_cols)
OutputDF_Digraph = PD.DataFrame(columns=df_cols2)
OutputDF_Digraph_Final = PD.DataFrame(columns=df_cols2)

"""loop = 0
while loop < len(FileName):
    data = PD.read_csv(r'Input/' + FileName[loop])
    User_ID = (FileName[loop]).split('_')[0]
    gender = demographic_data.loc[demographic_data["User ID"] == int(User_ID)]["Gender"].values[0]
    if gender == 'M':
        gender = 1
    else:
        gender = 0
    dept = demographic_data.loc[demographic_data["User ID"] == int(User_ID)]["Major/Minor"].values[0]
    age = demographic_data.loc[demographic_data["User ID"] == int(User_ID)]["Age"].values[0]

    MostOccurring_KeyList = data['key'].value_counts().head(5).index.tolist()
    # print(MostOccurring_KeyList[0])
    # *** Moving Window Logic ***
    maxLimit = len(data)
    base = 0
    high = 100

    # ******
    df2 = data.head(maxLimit)
    #print(df2)
    #rows = len(df2)
    while high <= maxLimit:
        df3 = data.iloc[base:high]
        isInitial = True
        errCount = 0
        base += 50
        high += 50
        rows = len(df3)
        # Digraphs - for the limit find the most recurring pair of keys
        # Then find their PP, RR, PR, RP Time
        r = 0
        KeyPressDifference = 0
        Typing_Speed = 0
        for value in MostOccurring_KeyList:
            df4 = df3.loc[df3['key'] == value].index.values
            df5 = data.iloc[df4] #[0]
            num = 0
            Counter = 0
            Totaldifference = 0
            DiffEachLoop = py.empty(int(len(df4)/2))
            TotaldifferenceSTD = 0
            for i in df4:
                if num < len(df4) and (num + 1) < len(df4):
                    Time_at_0 = df5.iloc[num]["time"]
                    num += 1
                    Time_at_1 = df5.iloc[num]["time"]
                    delta = datetime.strptime(Time_at_1,"%Y-%m-%d %H:%M:%S.%f") - datetime.strptime(Time_at_0, "%Y-%m-%d %H:%M:%S.%f")
                    Totaldifference += delta.total_seconds() * 1000
                    DiffEachLoop[Counter] = delta.total_seconds() * 1000
                    Counter += 1
                num += 1
            if len(df4) > 0:
                Totaldifference = Totaldifference/len(df4/2) # number of times key pressed eg 0,1, 0,1 = 2 times
                if len(DiffEachLoop) > 0:
                    TotaldifferenceSTD = py.std(DiffEachLoop)

            OutputDF.loc[len(OutputDF.index)] = [value, User_ID, dept, age, Totaldifference, TotaldifferenceSTD]
        #Digraph part
        # loop - row 1 with all other rows
        # loop the result and find diff of nth and n+2 rows
        while r < rows-2:
            KeyPress_time_k = df3.iloc[r]["time"]
            KeyRelease_time_k = df3.iloc[r+1]["time"]

            r += 2
            KeyPress_time_k_1 = df3.iloc[r]["time"]
            KeyRelease_time_k_1 = df3.iloc[r+1]["time"]
            KeyPressDifference = (datetime.strptime(KeyPress_time_k_1, "%Y-%m-%d %H:%M:%S.%f") -
                                  datetime.strptime(KeyPress_time_k, "%Y-%m-%d %H:%M:%S.%f")).total_seconds() * 1000
            KeyReleaseDifference = (datetime.strptime(KeyRelease_time_k_1, "%Y-%m-%d %H:%M:%S.%f") -
                                    datetime.strptime(KeyRelease_time_k, "%Y-%m-%d %H:%M:%S.%f")).total_seconds() * 1000
            KeyPressReleaseDifference = (datetime.strptime(KeyPress_time_k_1, "%Y-%m-%d %H:%M:%S.%f") -
                                         datetime.strptime(KeyRelease_time_k, "%Y-%m-%d %H:%M:%S.%f")).total_seconds() * 1000
            KeyReleasePressDifference = (datetime.strptime(KeyRelease_time_k_1, "%Y-%m-%d %H:%M:%S.%f") -
                                         datetime.strptime(KeyPress_time_k, "%Y-%m-%d %H:%M:%S.%f")).total_seconds() * 1000

            OutputDF_Digraph.loc[len(OutputDF_Digraph.index)] = [str(df3.iloc[r-2]["key"]) + "-" + str(df3.iloc[r]["key"]),
                        KeyPressDifference, KeyReleaseDifference, KeyPressReleaseDifference, KeyReleasePressDifference, 0, 0, 0, 0, 0, 0,
                                                                 gender]
        #print(OutputDF_Digraph)
        xcv = 0
        MostOccurring_DigraphList = json.loads(OutputDF_Digraph['Digraph'].value_counts().head(5).to_json())
        for key, value in MostOccurring_DigraphList.items():
            DigraphArray = py.where(OutputDF_Digraph["Digraph"] == key)
            DigraphArray = py.array(DigraphArray)
            DigraphSubset = OutputDF_Digraph.iloc[DigraphArray[0]]
            PP_Mean = DigraphSubset['K_PP_Time'].sum() / value
            PP_STD = py.std(DigraphSubset['K_PP_Time'])

            RR_Mean = DigraphSubset['K_RR_Time'].sum() / value
            RR_STD = py.std(DigraphSubset['K_RR_Time'])

            PR_Mean = DigraphSubset['K_PR_Time'].sum() / value
            PR_STD = py.std(DigraphSubset['K_PR_Time'])

            RP_Mean = DigraphSubset['K_RP_Time'].sum() / value
            RP_STD = py.std(DigraphSubset['K_RP_Time'])
            if isInitial:
                dig = DigraphSubset[0:len(DigraphSubset)]['Digraph']
                speed = py.where(DigraphSubset[0:len(DigraphSubset)]['K_PP_Time'] <= (DigraphSubset['K_PP_Time'] + 1000000))
                for ig in dig:
                    if ig.__contains__('SPACE') | ig.__contains__('BACKSPACE'):
                        typeSpeed = len(speed[0])
                        errCount += 1
                isInitial = False
            else:
                errCount = errCount
                typeSpeed = typeSpeed
            OutputDF_Digraph_Final.loc[len(OutputDF_Digraph_Final.index)] = [OutputDF_Digraph.iloc[xcv]['Digraph'], PP_Mean, RR_Mean,
                                                                             PR_Mean, RP_Mean, PP_STD, RR_STD, PR_STD, RP_STD, typeSpeed, errCount, gender]
            xcv += 1
        # print(OutputDF_Digraph_Final)
        # OutputDF_Digraph_Final.to_csv('Output.csv')

    #result = PD.concat([OutputDF, OutputDFTouch], axis=1)
    result = PD.concat([OutputDF, OutputDF_Digraph_Final], axis=1)
    print(FileName[loop])
    loop = loop + 1

result.to_csv('OutputMobile_All_116.csv')"""

result = PD.read_csv('OutputMobile_All_116.csv')
#result = result.loc[result['User_ID'] <= 85]
# print(result)
result = result.drop(['User_ID'], axis=1)
result = result.drop(['Dept'], axis=1)
result = result.drop(['Key'], axis=1)
result = result.drop(['Digraph'], axis=1)

for x in ['K_PP_Time', 'K_PR_Time', 'K_RP_Time']:
    q75, q25 = py.percentile(result.loc[:, x], [75, 25])
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)

    result.loc[result[x] < min, x] = py.nan
    result.loc[result[x] > max, x] = py.nan

result = result.dropna(axis=0)

x = result.iloc[:, : -1]
y= result.iloc[:, -1]
#y=result['L1_Gender']  # Labels
print("Here")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Perform PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_
explained_variance_cumulative = py.cumsum(explained_variance)

# Plot explained variance as a scree plot
plt.plot(range(1, len(explained_variance) + 1), explained_variance_cumulative, 'bo-', linewidth=2)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()

# Define the number of folds for cross-validation
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_pca, y_train)

# Combine the resampled data and labels into a single dataframe
data_resampled = PD.DataFrame(X_train_smote, columns=["pca_1", "pca_2", "pca_3", "pca_4", "pca_5", "pca_6", "pca_7", "pca_8", "pca_9", "pca_10"])
data_resampled["label"] = y_train_smote

# Save the resampled data to a CSV file
#data_resampled.to_csv("Data_Resampled_Desktop_PCA_110_10.csv", index=False)

# Define the number of folds for cross-validation
n_folds = 8

# Create the StratifiedKFold object
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Initialize lists to store the evaluation metrics for each fold
accuracy_scores = []
f1_scores = []
far_scores = []
frr_scores = []

# Perform cross-validation
for train_index, val_index in skf.split(X_train_smote, y_train_smote):
    # Split the data into training and validation sets for the current fold
    X_train_fold, X_val_fold = X_train_smote[train_index], X_train_smote[val_index]
    y_train_fold, y_val_fold = y_train_smote[train_index], y_train_smote[val_index]

    # Train the decision tree classifier on the training data of the current fold
    dtc = RandomForestClassifier(n_estimators=10, max_depth= 3)
    dtc.fit(X_train_fold, y_train_fold)

    # Predict on the validation set
    y_val_pred = dtc.predict(X_val_fold)

    # Calculate the evaluation metrics for the current fold
    accuracy_scores.append(accuracy_score(y_val_fold, y_val_pred))
    f1_scores.append(f1_score(y_val_fold, y_val_pred))

    # Calculate FAR and FRR for the current fold
    threshold = 0.5
    y_val_pred_prob = dtc.predict_proba(X_val_fold)[:, 1]
    y_val_pred_threshold = (y_val_pred_prob >= threshold).astype(int)
    far_scores.append(1 - precision_score(y_val_fold, y_val_pred_threshold))
    frr_scores.append(1 - recall_score(y_val_fold, y_val_pred_threshold))

# Calculate the average and standard deviation of the evaluation metrics across all folds
avg_accuracy = py.mean(accuracy_scores)
std_accuracy = py.std(accuracy_scores)

avg_f1 = py.mean(f1_scores)
std_f1 = py.std(f1_scores)

avg_far = py.mean(far_scores)
std_far = py.std(far_scores)

avg_frr = py.mean(frr_scores)
std_frr = py.std(frr_scores)

# Print the average and standard deviation of the evaluation metrics
print("Average Accuracy:", avg_accuracy)
print("Standard Deviation of Accuracy:", std_accuracy)

print("Average F1 Score:", avg_f1)
print("Standard Deviation of F1 Score:", std_f1)

print("Average False Acceptance Rate (FAR):", avg_far)
print("Standard Deviation of FAR:", std_far)

print("Average False Rejection Rate (FRR):", avg_frr)
print("Standard Deviation of FRR:", std_frr)

avg_accuracy_percent = avg_accuracy * 100
std_accuracy_percent = std_accuracy * 100
# Set the width of the bars
bar_width = 0.8

# Plotting the bar graph for accuracy with standard deviation
fig, ax = plt.subplots()
rects1 = ax.bar(0, avg_accuracy_percent, bar_width, label='Accuracy', yerr=std_accuracy_percent, capsize=10)

# Set the x-axis ticks and labels
ax.set_xticks([0])
ax.set_xticklabels(["Accuracy"])

# Set the y-axis label
ax.set_ylabel("Accuracy")

# Set the plot title
ax.set_title("Accuracy with Standard Deviation")

# Add a legend
ax.legend()

# Show the plot
plt.show()

# Perform SMOTE
"""smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_pca, y_train)

# Combine the resampled data and labels into a single dataframe
data_resampled = PD.DataFrame(X_train_smote, columns=["pca_1", "pca_2"])
data_resampled["label"] = y_train_smote

# Save the resampled data to a CSV file
data_resampled.to_csv("data_resampled_2.csv", index=False)
# Train decision tree classifier
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train_smote, y_train_smote)

# Predict on test set
y_pred = dtc.predict(X_test_pca)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# Calculate False Acceptance Rate (FAR)
# FAR is the percentage of genuine instances that are incorrectly classified as positive
# You need to define a threshold to determine the positive class based on the predicted probabilities
threshold = 0.5  # Adjust the threshold based on your problem
y_pred_prob = dtc.predict_proba(X_test_pca)[:, 1]  # Get the predicted probabilities for the positive class
y_pred_threshold = (y_pred_prob >= threshold).astype(int)  # Convert probabilities to binary predictions based on the threshold
far = 1 - precision_score(y_test, y_pred_threshold)
print("False Acceptance Rate (FAR):", far)

# Calculate False Rejection Rate (FRR)
# FRR is the percentage of impostor instances that are incorrectly classified as negative
# In this case, impostor instances can be considered as instances of the negative class
frr = 1 - recall_score(y_test, y_pred_threshold)

print("False Rejection Rate (FRR):", frr)"""

"""X_train, X_test, y_train, y_test = train_test_split(X_train_smote, y_train_smote, test_size=0.2)

# Create a logistic regression model and fit it to the data
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate the model on the test data
score = clf.score(X_test, y_test)
print("Accuracy:", score)

# Predict on test set
y_pred = clf.predict(X_test)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# Calculate False Acceptance Rate (FAR)
# FAR is the percentage of genuine instances that are incorrectly classified as positive
# You need to define a threshold to determine the positive class based on the predicted probabilities
threshold = 0.5  # Adjust the threshold based on your problem
y_pred_prob = clf.predict_proba(X_test)[:, 1]  # Get the predicted probabilities for the positive class
y_pred_threshold = (y_pred_prob >= threshold).astype(int)  # Convert probabilities to binary predictions based on the threshold
far = 1 - precision_score(y_test, y_pred_threshold)
print("False Acceptance Rate (FAR):", far)

# Calculate False Rejection Rate (FRR)
# FRR is the percentage of impostor instances that are incorrectly classified as negative
# In this case, impostor instances can be considered as instances of the negative class
frr = 1 - recall_score(y_test, y_pred_threshold)

print("False Rejection Rate (FRR):", frr)
# Train SVM classifier
svm = SVC(kernel='linear', C=1, gamma='auto')
svm.fit(X_train_smote, y_train_smote)

# Predict on test set
y_pred = svm.predict(X_test_pca)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)"""
