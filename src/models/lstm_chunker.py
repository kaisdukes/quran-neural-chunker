from typing import List

from pandas import DataFrame
import torch
from torch import nn
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from ..data import load_data
from ..chunks.preprocessor import preprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_length = 128


class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        packed_output, _ = self.lstm(x, (h0, c0))

        # Unpack the output before passing through the linear layer
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)

        # Manually pad the sequences to max_length
        if output.size(1) < max_length:
            output = nn.functional.pad(output, (0, 0, 0, max_length - output.size(1)))

        out = self.fc(output)
        return out


class QuranDataset(Dataset):
    def __init__(self, verses, labels):
        self.verses = verses
        self.labels = labels

    def __len__(self):
        return len(self.verses)

    def __getitem__(self, index):
        verse = self.verses[index]
        label = self.labels[index]
        length = len(verse)

        # padding
        if length < max_length:
            verse.extend([[0]*len(verse[0])] * (max_length - length))  # add 0 padding
            label.extend([0] * (max_length - length))  # add 0 padding

        return torch.tensor(verse, dtype=torch.float32), torch.tensor(label), length


def get_verses(df: DataFrame):

    le = LabelEncoder()
    df['encoded_punctuation'] = le.fit_transform(df['punctuation'])

    X = df[['token_number', 'pause_mark', 'irab_end', 'verse_end', 'encoded_punctuation']]
    y = df['chunk_end']

    verses: List[List[int]] = []
    labels: List[int] = []

    for _, group in df.groupby(['chapter_number', 'verse_number']):
        verse = group[X.columns].values.tolist()
        label = group[y.name].tolist()

        verses.append(verse)
        labels.append(label)

    train_verses, test_verses, train_labels, test_labels = train_test_split(verses, labels, test_size=0.10, random_state=42)
    return train_verses, test_verses, train_labels, test_labels


def pack_labels(labels):
    lengths = [len(label) for label in labels]
    max_len = max(lengths)
    labels_padded = [torch.cat([label, torch.zeros(max_len - len(label))]) for label in labels]
    return torch.stack(labels_padded)


def train_and_test():
    df = load_data()
    preprocess(df)

    df.fillna(0, inplace=True)

    input_size = 5
    hidden_size = 128
    num_layers = 2
    output_size = 2
    num_epochs = 15
    batch_size = 64
    learning_rate = 0.001

    train_verses, test_verses, train_labels, test_labels = get_verses(df)

    training_data = QuranDataset(train_verses, train_labels)
    testing_data = QuranDataset(test_verses, test_labels)

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # train
    model.train()
    for epoch in range(num_epochs):
        for i, (verses, labels, lengths) in enumerate(train_loader):
            verses = verses.to(device)
            labels = labels.to(device)

            # forward pass
            raw_outputs = model(verses, lengths)
            labels = labels.view(-1)  # reshape labels to be a 1D tensor
            loss = criterion(raw_outputs.view(-1, output_size), labels)

            # backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

        # test
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for verses, labels, lengths in test_loader:
                verses = verses.to(device)
                labels = labels.to(device)

                raw_outputs = model(verses, lengths)
                _, predicted = torch.max(raw_outputs.data, 2)  # get the predicted labels
                correct += (predicted.view(-1) == labels.view(-1)).sum().item()
                total += lengths.sum().item()

            print(f'Correct: {correct}')
            print(f'Total: {total}')
            print(f'Accuracy: {correct / total}')
