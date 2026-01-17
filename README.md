balanced_dataset_creation.py:-
It takes the initial dataset and uses smote to make the data balanced, then the balanced file is saved in the same directory.

sampling.py:-
I am sampling it with RandomUnderSampler,RandomUnderSampler,SMOTE,NearMiss,SMOTETomek samplers first i tried with balanced 
dataset but it gave the same samples everytime so i switched to the original dataset.

models.py:-
using
M1:-LogisticRegression
M2:-DecisionTree
M3:-RandomForest
M4:-KNN
M5:-SVM

compare_models.py:-
importing sampling.py and models.py both and creating an accuracy matrix for each model tested on each sample
and these are the best ones on each sample.

M1_LogisticRegression → Sampling3 (91.61%)
M2_DecisionTree → Sampling2 (99.35%)
M3_RandomForest → Sampling2 (99.35%)
M4_KNN → Sampling4 (98.06%)
M5_SVM → Sampling3 (96.13%)
