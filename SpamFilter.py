from Preprocessing import Preprocessing
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

# training data preprocessing
train_processor = Preprocessing()
train_processor.set_directory('train_dir')
train_labels_n = train_processor.get_emails_size()

# test data preprocesssing
test_processor = Preprocessing()
test_processor.set_directory('test_dir')
test_labels_n = test_processor.get_emails_size()

# prepare vectors
train_labels = np.zeros(train_labels_n)
test_labels = np.zeros(test_labels_n)
train_labels[351: train_labels_n] = 1     # this is spam emails and the rest is not
test_labels[130: test_labels_n] = 1       # this is spam emails and the rest is not

# extract features
train_features = train_processor.build_sparse_feat_matrix()
test_features = test_processor.build_sparse_feat_matrix()

# train the model using SVM with scikit learn
svm_classifier = LinearSVC()
svm_classifier.fit(train_features, train_labels)

# predict the models results
result = svm_classifier.predict(test_features)
print(confusion_matrix(test_labels, result))
