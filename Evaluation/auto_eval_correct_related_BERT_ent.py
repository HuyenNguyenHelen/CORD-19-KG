# import pandas as pd
# from pytorch_pretrained_bert import BertModel
# from torch import nn
# import BertBinaryClassifier
from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output
import tarfile
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import tree             # tree.DecisionTreeClassifier()
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import svm #clf = svm.SVC(decision_function_shape='ovo')
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
# !pip install imbalanced-learn
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import os

FILE_NAME = 'all'
UPSAMPLE = True
in_path = r"/home/junhua/huyen/CORD-19-KG/Evaluation/groundtruth/annotated-data/KG_EVAL_entities_types/"
out_path = r'/home/junhua/huyen/CORD-19-KG/Evaluation/result/KG_eval/TEST_Entities_'

####### Define helper functions ##################

def open_file(file_name):
    with open(in_path + file_name + '.csv', 'r', encoding='utf-8') as f:
        df = pd.read_csv(f)
    df.columns = ['Unnamed: 0', 'paragraph', 'label']
    return df


def divide_data(dataset):
    """
    Shuffle the dataset, and divide the dataset into 10 parts
    """
    # shuffle data
    dataset = dataset.sample(frac=1, axis=1, random_state = 42).reset_index(drop=True)
    # divide data
    subsets = np.array_split(dataset, 10)
    divided_data = []
    for i in range(len(subsets)):
        data = subsets[:i + 1]
        data_concat = pd.concat(data)
        divided_data.append(data_concat)
    return divided_data


# Printing model performance
def printing_eval_scores(y_true, y_pred):
    print('accuracy score: {}'.format(sklearn.metrics.accuracy_score(y_true, y_pred)))
    print(
        'precision score: {}'.format(sklearn.metrics.precision_score(y_true, y_pred, average='macro', zero_division=1)))
    print('recall score: {}'.format(sklearn.metrics.recall_score(y_true, y_pred, average='macro', zero_division=1)))
    print('F1 score: {}'.format(f1_score(y_true, y_pred, average='macro', zero_division=1)))
    print('\nConfusion Matrix:\n', confusion_matrix(y_true, y_pred))
    print('\n', classification_report(y_true, y_pred))


# Get the measurements of ROC curve for each model
def get_roc_cuve(y_true, y_pred):
    # Get arrays of FPR and recall using roc_curve
    FPR, recall, threshold = sklearn.metrics.roc_curve(y_true, y_pred)

    # Get testing accuracy:
    acc = accuracy_score(y_test, y_pred)

    # Get testing macro-f1:
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

    # Get auc score
    auc = sklearn.metrics.auc(FPR, recall)
    roc = {'fpr': FPR, 'tpr': recall, 'auc': auc, 'accuracy': acc, 'macro-F1': f1}
    return roc


def graph_multi_ROC(rocs):
    # Set color for each model
    colors = {'LGBM': 'lightcoral', 'LR': 'darkorange', 'SVM': 'lime', 'NB': 'steelblue',
              'XGB': 'purple', 'DT': 'magenta', 'RF': 'deeppink', 'KNN': 'darkturquoise',
              'BERT': 'darkred', 'GPT': 'blue'}
    # Set marker for each model
    markers = {'LGBM': '1--', 'LR': 'v--', 'SVM': '^--', 'XGB': '*--', 'DT': 'o--', 'RF': '+--', 'KNN': '.--',
               'NB': 'x--', 'BERT': '<--', 'GPT': '>--'}

    plt.figure(figsize=(9, 6))
    for model in rocs:
        plt.plot(rocs[model]['fpr'], rocs[model]['tpr'], markers[model], color=colors[model],
                 label=model + ' - AUC=' + str(rocs[model]['auc'].round(3)))

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chances')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.ylabel('Recall')
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.legend(loc='lower right')
    plt.title('ROC Curves of all models')
    plt.show()

# Create BERT
from pytorch_pretrained_bert import BertModel
from torch import nn

class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tokens, masks=None):
        # First Layer
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)

        dropout_output = self.dropout(pooled_output)

        linear_output = self.linear(dropout_output)

        # output layer
        proba = self.sigmoid(linear_output)

        return proba

    def train_m(self, x, y, train_mask, epochs, batchsize):
        train_tokens_tensor = torch.tensor(x)
        train_y_tensor = torch.tensor(y.reshape(-1, 1)).float()
        train_masks_tensor = torch.tensor(train_mask)

        train_dataset = TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batchsize)

        param_optimizer = list(self.sigmoid.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        optimizer = Adam(self.bert.parameters(), lr=2e-5)
        for epoch_num in range(epochs):
            self.train()  # Training Flag
            train_loss = 0
            for step_num, batch_data in enumerate(train_dataloader):
                # Load batch on device memory
                token_ids, masks, labels = tuple(t.to(device) for t in batch_data)

                # Get the output of the model for provided input
                logits = self(token_ids, masks)

                # Loss function
                loss_func = nn.BCELoss()

                # Calculate Loss
                batch_loss = loss_func(logits, labels)
                train_loss += batch_loss.item()

                # backpropagate the error
                self.zero_grad()
                batch_loss.backward()

                # Update the Weights of the Model
                clip_grad_norm_(parameters=self.parameters(), max_norm=1.0)
                optimizer.step()

                clear_output(wait=True)
                print('Epoch: ', epoch_num + 1)
                print("\r" + "{0}/{1} loss: {2} ".format(step_num, len(train_labels) / batchsize,
                                                         train_loss / (step_num + 1)))


############ Processing data ############################
data = open_file(FILE_NAME)
# Splitting the data into training (80%) and test set(20%)
X = data['paragraph']
y = data['label']
_X_train, X_test, _y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, shuffle=True, stratify=y)

train_data = pd.concat([_X_train, _y_train], axis=1)
print(train_data.shape)
train_subsets = divide_data(train_data)


######## Train BERT on different portions of training set #############
for i,df in enumerate(train_subsets):
    X_train, y_train = df['paragraph'], df['label']
    print('Shapes of X_train, y_train: ', _X_train.shape, _y_train.shape)
    print('Shapes of X_test, y_test: ', X_test.shape, y_test.shape)
    ############ Encoding data to BERT ##########################
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # get max len in tokenized train text to set the tokens length in the next step
    MAX_LEN = max(map(len, X_train))  # can do len(max(X_train, key=len)) also
    print('MAX LEN of training sentence is:', MAX_LEN, '\nMAX LEN > 512 is ', MAX_LEN > 512)

    # Update MAX LEN if it's > 512, set it to be 225
    ## 512 is is the maximum seq len of BERT_BASE. But we cannot allow the seq len to be 512 since we'll run out of GPU memory --> Use max len of 225
    MAX_LEN = 225 if MAX_LEN > 512 else MAX_LEN

    # Convert to tokens using tokenizer
    train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[: MAX_LEN] + ['[SEP]'], X_train.to_list()))
    test_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[: MAX_LEN] + ['[SEP]'], X_test.to_list()))
    print('Number of Training Sequences:', len(train_tokens), '\nNumber of Testing Sequences:', len(test_tokens))

    # Following is to convert List of words to list of numbers. (Words are replaced by their index in dictionar)
    train_tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, train_tokens)), maxlen=MAX_LEN,
                                     truncating="post", padding="post", dtype="int")
    test_tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, test_tokens)), maxlen=MAX_LEN,
                                    truncating="post", padding="post", dtype="int")

    # Upsampling option
    if UPSAMPLE:
        # Upsmapled training data for BERT
        oversampler = SMOTE(random_state=42)
        train_tokens_ids, train_labels = oversampler.fit_resample(train_tokens_ids, y_train)
    else:
        train_labels = y_train

    # Mask the paddings with 0 and words with 1
    train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
    test_masks = [[float(i > 0) for i in ii] for ii in test_tokens_ids]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device

    ############### Train BERT ###############################
    # Initiate BERT Classifier using cuda
    bert_clf = BertBinaryClassifier()
    bert_clf = bert_clf.cuda()

    EPOCHS = 3
    BATCH_SZ = 32

    # Train BERT NLP
    bert_clf.train_m(train_tokens_ids, train_labels.to_numpy(), train_masks, EPOCHS, BATCH_SZ)

    ########### Test BERT #######################
    ## Converting test token ids, test labels and test masks to a tensor and the create a tensor dataset out of them.
    # Convert token ids to tensor
    test_tokens_tensor = torch.tensor(test_tokens_ids)

    # Convert labels to tensors
    test_y_tensor = torch.tensor(y_test.to_numpy().reshape(-1, 1)).float()

    # Convert to tensor for maks
    test_masks_tensor = torch.tensor(test_masks)

    # Load Token, token mask and label into Dataloader
    test_dataset = TensorDataset(test_tokens_tensor, test_masks_tensor, test_y_tensor)

    # Define sampler
    test_sampler = SequentialSampler(test_dataset)

    # Define test data loader
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=16)

    bert_clf.eval()  # Define eval
    bert_predicted = []  # To Store predicted result
    all_logits = []  # Predicted probabilities that is between 0 to 1 is stored here

    with torch.no_grad():
        for step_num, batch_data in enumerate(test_dataloader):
            # Load the batch on gpu memory
            token_ids, masks, labels = tuple(t.to(device) for t in batch_data)

            # Calculate ouput of bert
            logits = bert_clf(token_ids, masks)

            # Get the numpy logits
            numpy_logits = logits.cpu().detach().numpy()  # Detach from the GPU memory

            # Using the threshold find binary
            bert_predicted += list(numpy_logits[:, 0] > 0.5)  # Threshold conversion
            all_logits += list(numpy_logits[:, 0])

    rocs = {}
    # Get ROC curve measurements
    rocs['BERT'] = get_roc_cuve(y_test, bert_predicted)

    # Prin performance
    print('----------------------------BERT performance---------------------------')
    printing_eval_scores(y_test, bert_predicted)

    # Export performance to a txt file

    # Set file name according to Upsampling option
    if UPSAMPLE == True:
        name = 'BERT_SMOTE'
    else:
        name = 'BERT'

    txtfile = open(out_path + name + '.txt', 'a+')
    txtfile.write('subset%s_'%i + FILE_NAME + '_' + name + '=' + str(rocs['BERT']) + '\n')

    txtfile.close()
    graph_multi_ROC(rocs)

