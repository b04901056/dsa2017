This the Readme file of Qualcomm interview task 
## Requirements

- Python 3.6
- Pytorch 0.3
- numpy
- matplotlib
- sklearn
- imblearn

## Instruction to run the code

DNN implementation
```
python .\train.py -train .\DataMining_IV_x2\DataSet_IV_Training.csv -test .\DataMining_IV_x2\DataSet_IV_Test.csv  -u 170 32 1
```

SVM implementation
```
python .\svm.py .\DataMining_IV_x2\DataSet_IV_Training.csv .\DataMining_IV_x2\DataSet_IV_Test.csv
```

Logistic Regression implementation
```
python .\logistic_regression.py .\DataMining_IV_x2\DataSet_IV_Training.csv .\DataMining_IV_x2\DataSet_IV_Test.csv
```

Decision Tree implementation
```
python .\decision_tree.py .\DataMining_IV_x2\DataSet_IV_Training.csv .\DataMining_IV_x2\DataSet_IV_Test.csv
```

t1_p1: True  positive t0_p1:  False negative
t1_p0: False positive t0_p0:   True negative

The image represents the result of classification on testing data after applying PCA to reduce dimension to 2.