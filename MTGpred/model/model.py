from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MTGpred.utils.database import get_all_matches_ids
from MTGpred.model.dataset import MatchesDataset
from MTGpred.utils.mtgjson import load_cards_df
import random
from tqdm import tqdm

class DeckEncoder(nn.Module):
    def __init__(self):
        super(DeckEncoder,self).__init__()
        # Decks are 2D tensors of shape (1,100,768)
        self.conv1 = nn.Conv1d(1,16,3,padding="same")   # (16,100,768)
        self.maxpool1 = nn.MaxPool1d(2)                 # (16,50,384)
        self.conv2 = nn.Conv1d(16,32,3,padding="same")  # (32,50,384)
        self.maxpool2 = nn.MaxPool1d(2)                 # (32,25,192)
        self.conv3 = nn.Conv1d(32,64,3,padding="same")  # (64,25,192)
        self.maxpool3 = nn.MaxPool1d(2)                 # (64,12,96)
        self.conv4 = nn.Conv1d(64,128,3,padding="same") # (128,12,96)
        self.maxpool4 = nn.MaxPool1d(2)                 # (128,6,48)
        self.conv5 = nn.Conv1d(128,256,3,padding="same")# (256,6,48)
        self.maxpool5 = nn.MaxPool1d(2)                 # (256,3,24)
        self.conv6 = nn.Conv1d(256,512,3,padding="same")# (512,3,24)
        self.maxpool6 = nn.MaxPool1d(3)                 # (512,1,8)
        self.flatten = nn.Flatten()                     # (4096)
        self.fc1 = nn.Linear(4096,512)                  # (512)
        self.fc2 = nn.Linear(512,256)                   # (256)

    def forward(self,deck):
        deck = deck.unsqueeze(0)

        # deck is a 2D tensor of shape (1,100,768)
        deck = self.conv1(deck)
        deck = F.relu(deck)
        deck = self.maxpool1(deck)

        deck = self.conv2(deck)
        deck = F.relu(deck)
        deck = self.maxpool2(deck)

        deck = self.conv3(deck)
        deck = F.relu(deck)
        deck = self.maxpool3(deck)

        deck = self.conv4(deck)
        deck = F.relu(deck)
        deck = self.maxpool4(deck)

        deck = self.conv5(deck)
        deck = F.relu(deck)
        deck = self.maxpool5(deck)

        deck = self.conv6(deck)
        deck = F.relu(deck)
        deck = self.maxpool6(deck)

        deck = self.flatten(deck)

        deck = self.fc1(deck)
        deck = F.relu(deck)

        deck = self.fc2(deck)
        deck = F.relu(deck)

        return deck

class WinnerPredictor(nn.Module):
    def __init__(self):
        super(WinnerPredictor,self).__init__()
        self.encoder = DeckEncoder()
        self.final_classifier = nn.Linear(512,1)
    
    def forward(self,deck1,deck2):
        deck1 = self.encoder(deck1)
        deck2 = self.encoder(deck2)
        output = self.final_classifier(torch.cat((deck1,deck2),dim=1))
        winner = F.sigmoid(output)
        return winner

def accuracy(y_pred,y_true):
    return (y_pred == y_true).sum().item()/len(y_pred)

def train(split_ratio=0.8,batch_size=16,epochs=10,lr=0.001,cards_path = "data\AtomicCards.json",cuda=True):
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    # Load data
    all_matches_ids = get_all_matches_ids()
    random.shuffle(all_matches_ids)
    split = int(len(all_matches_ids)*split_ratio)
    train_matches_ids = all_matches_ids[:split]
    test_matches_ids = all_matches_ids[split:]

    cards_df = load_cards_df(cards_path)
    train_dataset = MatchesDataset(cards_df,train_matches_ids,device)
    test_dataset = MatchesDataset(cards_df,test_matches_ids,device)

    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

    # Train model
    model = WinnerPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print(f"======= EPOCH {epoch+1}/{epochs} =======")
        train_losses = []
        for batch in tqdm(train_dataloader,desc="TRAIN"):
            print(batch)
            input = batch[0].to(device)
            target = batch[1].to(device)
            optimizer.zero_grad()
            winner = model(input[:,0],input[:,1])
            loss = criterion(winner,target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        print(f"Train loss: {sum(train_losses)/len(train_losses)}")

        with torch.no_grad():
            test_losses = []
            accuracies = []
            for input,output in tqdm(test_dataloader,desc="TEST"):
                winner = model(input[:,0],input[:,1])
                loss = criterion(winner,output)
                test_losses.append(loss.item())
                accuracies.append(accuracy(winner,output > 0.5))
            print(f"Test loss: {sum(test_losses)/len(test_losses)}")
            print(f"Test accuracy: {sum(accuracies)/len(accuracies)}")

    # Save model
