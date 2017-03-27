from __future__ import unicode_literals
import pandas as pd
import nltk
from sklearn.metrics import log_loss
from scipy.optimize import minimize
# nltk.download()


def analyse_questions(dataframe):
    # Transform questions in vectors of words
    dataframe['tok1'] = dataframe.apply(lambda row: nltk.word_tokenize(str(row['question1']).decode('utf-8').lower()), axis=1)
    dataframe['tok2'] = dataframe.apply(lambda row: nltk.word_tokenize(str(row['question2']).decode('utf-8').lower()), axis=1)

    # Import stopwords to focus on important content of sentence.
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # Filter stopwords out of vectors
    dataframe['filtered1'] = dataframe['tok1'].apply(lambda x: [word for word in x if word not in stop_words])
    dataframe['filtered2'] = dataframe['tok2'].apply(lambda x: [word for word in x if word not in stop_words])

    nltk.tokenize.PunktSentenceTokenizer
    # Group variations of words together
    lemmatizer = nltk.stem.WordNetLemmatizer()
    dataframe['lemmatized1'] = dataframe["filtered1"].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
    dataframe['lemmatized2'] = dataframe["filtered2"].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])

    # Find common words between each couples of questions
    common = []
    for row in dataframe.index:
        common.append(set(dataframe['lemmatized1'][row]) & set(dataframe['lemmatized2'][row]))
    dataframe['common'] = common

    # Number of words per question (average over couple of questions)
    avgqlength = []
    for row in dataframe.index:
        avgqlength.append((len(dataframe['lemmatized1'][row]) + len(dataframe['lemmatized2'][row])) / 2)
    dataframe['avgqlenth'] = avgqlength

    # Ratio of common words to average number of words in the questions
    commonratio = []
    for row in dataframe.index:
        if dataframe['avgqlenth'][row] == 0:
            commonratio.append(0)
        else:
            commonratio.append(float(len(dataframe['common'][row])) / float(dataframe['avgqlenth'][row]))
    dataframe['commonratio'] = commonratio

    return dataframe['commonratio']

# read file
train = pd.read_csv("/home/laurent/R_Projects/Quora/train.csv", index_col="id")
train['commonratio'] = analyse_questions(train)

# Optimize weight compared to the number of duplicates in the dataset
GLOBAL_MEAN = np.mean(train['is_duplicate'])
def minimize_train_log_loss( W ):
    train['prediction'] = GLOBAL_MEAN + train['commonratio'] * W[0] + W[1]
    score = log_loss( train['is_duplicate'], train['prediction'] )

res = minimize(minimize_train_log_loss, [0.00,  0.00], method='Nelder-Mead', tol=1e-4, options={'maxiter': 400})
W = res.x
print('Best weights: ', W)


test = pd.read_csv("/home/laurent/R_Projects/Quora/test.csv")
test['commonratio'] = analyse_questions(test)
test['weighted'] = GLOBAL_MEAN + test['commonratio'] * W[0] + W[1]
dupes = []
for row in test.index:
    dupes.append(int(round(test['weighted'][row])))
test['is_duplicate'] = dupes
test[['test_id', 'is_duplicate']].to_csv('/home/laurent/R_Projects/Quora/submission.csv', header=True, index=False)
