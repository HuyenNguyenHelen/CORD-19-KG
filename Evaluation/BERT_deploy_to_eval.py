import pandas as pd
import torch
import transformers
# from pytorch_pretrained_bert import BertModel
from transformers import BertModel, BertTokenizer, BertConfig, BertForTokenClassification

import torch.nn as nn

# from torch import nn
# from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
# from IPython.display import clear_output


# Define BERT model
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


# Create main function
def main(saved_model_path, data_path, correct_compreh = 'correct_ent' ):
  """
  correct_compreh: 'correct_ent', 'correct_trip', 'compreh'
  """
  # Loading data
  
  with open(data_path, 'r') as f:
    data_test = pd.read_csv(f)
  
  if correct_compreh == 'correct_ent':
    X_test = data_test['subject'].to_list() + data_test['object'].to_list()

    # Tokenizer 
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  
  # Update MAX LEN 
  MAX_LEN = 228 

  # Convert to tokens using tokenizer
  test_tokens  = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[: MAX_LEN] + ['[SEP]'], X_test))

  print( '\nNumber of Testing Sequences:', len(test_tokens) )
  # Following is to convert List of words to list of numbers. (Words are replaced by their index in dictionar)
  test_tokens_ids  = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, test_tokens)),  maxlen= MAX_LEN, truncating="post", padding="post", dtype="int")
  # Mask the paddings with 0 and words with 1
  test_masks = [[float(i > 0) for i in ii] for ii in test_tokens_ids]

  ## Converting test token ids, test labels and test masks to a tensor and the create a tensor dataset out of them.
  # Convert token ids to tensor 
  test_tokens_tensor = torch.tensor(test_tokens_ids)

  # Convert labels to tensors
  # test_y_tensor = torch.tensor(y_test.to_numpy().reshape(-1, 1)).float()

  # Convert to tensor for maks
  test_masks_tensor = torch.tensor(test_masks)

  # Load Token, token mask and label into Dataloader
  test_dataset = TensorDataset(test_tokens_tensor, test_masks_tensor)

  # Define sampler
  test_sampler = SequentialSampler(test_dataset)

  # Define test data loader
  test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=16)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  bert_clf = BertBinaryClassifier()
  bert_clf.load_state_dict(torch.load(saved_model_path),  strict=False)

  bert_clf.eval()     # Define eval
  bert_predicted = [] # To Store predicted result
  all_logits = []     # Predicted probabilities that is between 0 to 1 is stored here

  with torch.no_grad():
      for step_num, batch_data in enumerate(test_dataloader):

          # Load the batch on gpu memory
          token_ids, masks = tuple(t.to(device) for t in batch_data)

          # Calculate ouput of bert
          logits = bert_clf(token_ids, masks)

          # Get the numpy logits
          numpy_logits = logits.cpu().detach().numpy()  # Detach from the GPU memory
          
          # Using the threshold find binary 
          bert_predicted += list(numpy_logits[:, 0] > 0.5)  # Threshold conversion
          # all_logits += list(numpy_logits[:, 0])
  print(bert_predicted)



if __name__=='__main__':
  main(saved_model_path = '/home/huyen/CORD-19-KG/Evaluation/DATA-QUALITY_EVAL_SAVE_MODEL/all.h5', 
       data_path = '/home/huyen/CORD-19-KG/Data/all-final-cleaned-triple3-10sets/subset_9.csv'  )