import numpy as np
import pandas as pd


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from warnings import filterwarnings

def evaluate(encoder, train_df, test_df, result_df, model_name):
    """ Evaluate the learned representation of the encoder using a logistic regression
    """
    filterwarnings('ignore')
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()
    for i in range(len(train_df_copy)):
        train_df_copy.at[i, 'text'] = encoder(train_df_copy.at[i, 'text'].unsqueeze(0)).detach().numpy()

    
    X_train = np.array(train_df_copy['text'].tolist())
    X_train = np.squeeze(X_train, axis=1)
    y_train = np.array(train_df_copy['coarse_label'].tolist())

    clf = LogisticRegression(random_state=0).fit(X_train, y_train)

    clf.fit(X_train, y_train)

    for i in range(len(test_df)):
        test_df_copy.at[i, 'text'] = encoder(test_df_copy.at[i, 'text'].unsqueeze(0)).detach().numpy()
    
    X_test = np.array(test_df_copy['text'].tolist())
    X_test = np.squeeze(X_test, axis=1)
    y_test = np.array(test_df_copy['coarse_label'].tolist())

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    result_df = pd.concat([result_df, pd.DataFrame([[model_name, train_acc, test_acc]], columns=['model_name', 'train_accuracy', 'test_accuracy'])], ignore_index=True)

    return result_df