import torch
import random
from random import shuffle
random.seed(11)


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(11)
    
    
class MyNeuralClassifier(nn.Module):
    """
    This is a neural classifier.
    """

    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim):
        """
        Note: if your model needs other parameters, feel free to change them.
        """
        super(MyNeuralClassifier, self).__init__()
        
        # TODO: implement your preferred architecture!
        self.hidden=nn.Linear(vocab_size,hidden_dim)
        self.output=nn.Linear(hidden_dim,output_dim)
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax(dim=1)
    def forward(self, input):
        # TODO: implement your preferred architecture!
        input=self.hidden(input)
        input=self.sigmoid(input)
        input=self.output(input)
        input=self.softmax(input)
        return input
        
    
def load_data(fname = 'BYOSC_data.csv'):
    """
    This function loads the data. The default value for fname assumes that
    the file with the data is in the same folder as this python file.
    If this isn't true, change the value of fname to the location of your data.
    """
    data = []
    for line in open(fname, encoding="utf8", errors="ignore"):
        # This only loads the columns that we care about.
        y, _, _, _, _, x = line[1:-2].strip().split('","')
        data.append((x, y))
    
    # This shuffles the dataset. shuffle() gives you the data in a random order.
    # However, we have set random.seed(11) above, so this should give the same
    # order every time you run this code.
    shuffle(data)
    
    # We will use the first 300 examples for training and the rest as our dev set.
    # (We will not use a test set; you can assume that would be held-out data.)
    train_set = data[:300]
    dev_set = data[300:]

    return train_set, dev_set
    

def make_feature_vectors(data, w2i):
    """
    This function takes text data and returns vectors with the indices of the words.
    """
    new_data = []

    for (x, y) in data:
        sentence = []
        for word in x.split(' '):
            if word in w2i:
                sentence.append(w2i[word])
            else:
                sentence.append(w2i["<UNK>"])

        new_data.append((torch.tensor(sentence), torch.tensor([int(int(y)/2)])))
    
    return new_data


def eval(model, data):
    """
    This function evaluates the model on the given data.
    It prints the accuracy.
    """
    # Set the model to evaluation mode; no updates will be performed.
    # (The opposite command is model.train().)
    model.eval()
    
    total = right = 0
    for (x, y) in data:
        # model(x) calls the forward() function of the model.
        # Here is the point where we manually add a softmax function.
        probs = F.softmax(model(x), dim=0)
        y_hat = torch.argmax(probs)
        if y_hat == y:
            right += 1
        total += 1
        
    print("Accuracy: " + str((right * 1.0)/total))


def print_params(model):
    """
    This function can be used to print (and manually inspect) the model parameters.
    """
    for name, param in model.named_parameters():
        print(name)
        print(param)
        print(param.grad)


def get_vocab(data):
    w2i = {'<UNK>': 0}

    for (x, y) in data:
        for word in x.split(' '):
            if word not in w2i:
                w2i[word] = len(w2i)

    return w2i
    

if __name__ == "__main__":
    """
    This is the entry point for the code.
    """
    # This loads the training and development set.
    train, dev = load_data()
    # This constructs the vocabulary.
    vocab = get_vocab(train)
    train = make_feature_vectors(train, vocab)
    dev = make_feature_vectors(dev, vocab)
    
    # This creates the classifier (with random parameters!).
    # TODO: substitute -1 by more reasonable values!
    model = MyNeuralClassifier(len(vocab),784,256,10)
    
    # The next 2 lines define the loss function and the optimizer.
    # SGD = stochastic gradient descent
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=.1)
    
    # This is a sanity check: what is the model performance on train and dev?
    # The model parameters are currently random, so the performance should be, too.
    eval(model, train)
    eval(model, dev)
    print()
        
    # The next part is the actual model training.
    epochs = 3  # how many times does the model see the entire training set?
    for i in range(epochs):
        print("Starting epoch " + str(i))
        # The next line ensures that model parameters are being updated.
        # (The opposite command is model.eval().)
        model.train()
        for (x, y) in train:
            model.zero_grad()
            # Compute the model's prediction for the current example.
            raw_scores = model(x)
            # Compute the loss from the prediction and the gold label.
            loss = loss_function(raw_scores.unsqueeze(0), y)
            # Compute the gradients.
            loss.backward()
            # Update the model; the parameters should change during this step.
            optimizer.step()
        
        # This is a second sanity check.
        # The model performance should increase after every epoch!
        eval(model, train)
        eval(model, dev)
        print()
        
    
