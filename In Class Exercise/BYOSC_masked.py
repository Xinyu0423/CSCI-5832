import random
from random import shuffle
random.seed(11)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(11)


class MyClassifier(nn.Module):
    """
    A simple logistic regression classifier.
    Note that the forward function doesn't use sigmoid/softmax.
    This is an implementation detail, sigmoid or softmax is added either by the loss function
    or manually later in the code.
    """

    def __init__(self, num_features, num_labels):
        super(MyClassifier, self).__init__()

        # TODO: substitute -1 with the correct value!
        self.linear = nn.Linear(num_features=3)
    
    def forward(self, input):
        return self.linear(input)
    
    
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
    This function takes text data and returns feature vectors.
    This is done by counting how many times each of the features (words by default) appear in the text.
    The features are expected in the form of a dictionary mapping each feature to an index.
    
    Example:
    Features: {"liked": 0, "happy": 1, "sad": 2}
    Text: "I liked the movie, I liked the songs, I liked the actors; I am very happy right now".
    Produced feature vector: [3, 1, 0]
    """
    new_data = []

    for (x, y) in data:
        sentence = [0.] * len(w2i)
        for word in x.split(' '):
            if word in w2i:
                sentence[w2i[word]] += 1
            else:
                sentence[w2i['<UNK>']] += 1
        # The last expression in the next line is the gold label y as given in the dataset.
        new_data.append((torch.tensor(sentence), torch.tensor([int(int(y)/2)])))
        
    return new_data


def eval(model, data):
    """
    This function evaluates the model on the given data.
    It should print the accuracy.
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
        
        # TODO: how do we get our accuracy?
    acc=torch.sum(total).float()/right.numel()    
    print("Accuracy: " + str((right * 1.0)/total))


def print_params(model):
    """
    This function can be used to print (and manually inspect) the model parameters.
    """
    for name, param in model.named_parameters():
        print(name)
        print(param)
        print(param.grad)


def get_features(data):
    """
    This function defines features used for classification.
    """
    w2i = {'<UNK>': 0, ':)': 1, ':(': 2, 'LOL': 3, ':D': 4, 'amazing': 5, 'sad': 6, 'happy': 7}

    return w2i
    

if __name__ == "__main__":
    """
    This is the entry point for the code.
    """
    # This loads the training and development set.
    train, dev = load_data()
    feature_dict = get_features(train)
    train = make_feature_vectors(train, feature_dict)
    dev = make_feature_vectors(dev, feature_dict)
    
    # This creates the classifier (with random parameters!).
    # TODO: substitute -1 with the correct value!
    model = MyClassifier(len(feature_dict), -1)
    
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
        
    
