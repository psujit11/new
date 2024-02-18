import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from scipy import sparse
import numpy as np
import re

# Load your train.csv dataset
train_df = pd.read_csv('./dataset/train.csv')

def feature_engineering(df, sparse=0): 
    
    # Comment length
    df['length'] = df.comment_text.apply(lambda x: len(x))
    

    # Capitalization percentage
    def pct_caps(s):
        return sum([1 for c in s if c.isupper()]) / (sum(([1 for c in s if c.isalpha()])) + 1)
    df['caps'] = df.comment_text.apply(lambda x: pct_caps(x))

    # Mean Word length 
    def word_length(s):
        s = s.split(' ')
        return np.mean([len(w) for w in s if w.isalpha()])
    df['word_length'] = df.comment_text.apply(lambda x: word_length(x))

    # Average number of exclamation points 
    df['exclamation'] = df.comment_text.apply(lambda s: len([c for c in s if c == '!']))

    # Average number of question marks 
    df['question'] = df.comment_text.apply(lambda s: len([c for c in s if c == '?']))
    
    # Normalize
    for label in ['length', 'caps', 'word_length', 'question', 'exclamation']:
        minimum = df[label].min()
        diff = df[label].max() - minimum
        df[label] = df[label].apply(lambda x: (x-minimum) / (diff))

    # Strip IP Addresses
    ip = re.compile('(([2][5][0-5]\.)|([2][0-4][0-9]\.)|([0-1]?[0-9]?[0-9]\.)){3}'
                    +'(([2][5][0-5])|([2][0-4][0-9])|([0-1]?[0-9]?[0-9]))')
    def strip_ip(s, ip):
        try:
            found = ip.search(s)
            return s.replace(found.group(), ' ')
        except:
            return s

    df.comment_text = df.comment_text.apply(lambda x: strip_ip(x, ip))
    
    return df

# def merge_features(comment_text, data, engineered_features):
#     new_features = sparse.csr_matrix(data[engineered_features].values)
#     if np.isnan(new_features.data).any():
#         new_features.data = np.nan_to_num(new_features.data)
#     return sparse.hstack([comment_text, new_features])

def merge_features(comment_text, data, engineered_features):
    comment_text_sparse = TfidfVectorizer().fit_transform(data['comment_text'])  # Assuming TfidfVectorizer for comment_text
    new_features = sparse.csr_matrix(data[engineered_features].values)
    
    if np.isnan(new_features.data).any():
        new_features.data = np.nan_to_num(new_features.data)
    
    return sparse.hstack([comment_text_sparse, new_features])


# Assuming df is your train dataset
train_df = feature_engineering(train_df)

# Load test.csv and test_label.csv
test_df = pd.read_csv('./dataset/test.csv')
test_labels_df = pd.read_csv('./dataset/test_labels.csv')

# Feature engineering for test set
test_df = feature_engineering(test_df)

# Merge features for training set
engineered_features = ['length', 'caps', 'word_length', 'question', 'exclamation']
X_train = merge_features(train_df['comment_text'], train_df, engineered_features)

# Merge features for test set
X_test = merge_features(test_df['comment_text'], test_df, engineered_features)

# Convert multi-labels to a binary matrix
from sklearn.preprocessing import MultiLabelBinarizer


mlb = MultiLabelBinarizer()
y_train_binary = mlb.fit_transform(y_train)

# Ensure X_train and y_train_binary have the same number of samples
min_samples = min(X_train.shape[0], y_train_binary.shape[0])
X_train = X_train[:min_samples]
y_train_binary = y_train_binary[:min_samples]

# Split the data
X_train, X_val, y_train_binary, y_val = train_test_split(X_train, y_train_binary, test_size=0.2, random_state=42)


# Print shapes for debugging
print("X_train shape:", X_train.shape)
print("y_train_binary shape:", y_train_binary.shape)

# Check if y_train_binary has only 2 dimensions
if len(y_train_binary.shape) != 2:
    y_train_binary = np.reshape(y_train_binary, (len(y_train_binary), -1))

# Print shapes again
print("Updated y_train_binary shape:", y_train_binary.shape)

# Now try fitting the model again
svm_model.fit(X_train, y_train_binary)
//
# Print shapes for debugging
print("X_train shape:", X_train.shape)
print("y_train_binary shape:", y_train_binary.shape)

# Check if y_train_binary has only 2 dimensions
if len(y_train_binary.shape) != 1:
    # Assuming y_train_binary is a NumPy array, extract the first column as the target
    y_train_binary = y_train_binary[:, 0]

# Print shapes again
print("Updated y_train_binary shape:", y_train_binary.shape)

# Now try fitting the model again
svm_model.fit(X_train, y_train_binary)

# Feature engineering for test set
test_df = feature_engineering(test_df)

# Merge features for test set
X_test = merge_features(test_df['comment_text'], test_df, engineered_features)

# Predictions on the test set
y_test_pred = svm_model.predict(X_test)

# Assuming y_test_labels_binary is the binary representation of test_labels_df
y_test_labels_binary = mlb.transform(test_labels_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])

# Accuracy on the test set
accuracy_test = np.mean(y_test_pred == y_test_labels_binary)
print("Test Accuracy:", accuracy_test)

# Display predictions for the test set
for comment, prediction, true_labels in zip(test_df['comment_text'], y_test_pred, y_test_labels_binary):
    print(f"Comment: {comment}\nPrediction: {prediction}\nTrue Labels: {true_labels}\n")
