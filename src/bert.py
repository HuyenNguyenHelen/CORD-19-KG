from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
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

if __name__ == '__main__':
    BertBinaryClassifier(nn.Module)
