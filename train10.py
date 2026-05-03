import os
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader

# ==============================
# CONFIG
# ==============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RAVDESS = r"C:\Users\cmaha\RAVDESS"
CREMA = r"C:\Users\cmaha\AudioWAV"
LAUGH = r"C:\Users\cmaha\laughter_only"
EHEHE = r"C:\Users\cmaha\ehehe_data\data"

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_mel(path):
    try:
        y, sr = librosa.load(path, sr=22050)
        if len(y) == 0:
            return None
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel = librosa.power_to_db(mel)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)

        if mel.shape[1] < 128:
            mel = np.pad(mel, ((0,0),(0,128-mel.shape[1])))
        else:
            mel = mel[:, :128]

        return mel.astype(np.float32)
    except:
        return None

# ==============================
# LOADERS
# ==============================
def load_data(path, label):
    X, y = [], []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".wav"):
                feat = extract_mel(os.path.join(root, f))
                if feat is not None:
                    X.append(feat)
                    y.append(label)
    return X, y

def load_ehehe_split():
    X, y = [], []
    for root, _, files in os.walk(EHEHE):
        for f in files:
            if f.endswith(".wav"):
                feat = extract_mel(os.path.join(root, f))
                if feat is not None:
                    X.append(feat)
                    y.append(1)  # laughter
    return train_test_split(X, y, test_size=0.9, random_state=SEED)

# ==============================
# DATASET
# ==============================
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X)).unsqueeze(1)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# ==============================
# MODEL
# ==============================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,2)
        )

    def forward(self,x):
        return self.net(x)

# ==============================
# TRAIN / EVAL
# ==============================
def train(model, loader, opt, loss_fn):
    model.train()
    total = 0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out,y)
        loss.backward()
        opt.step()
        total += loss.item()
    return total/len(loader)

def evaluate(model, loader):
    model.eval()
    preds, trues, probs = [], [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE)
            out = model(x)
            p = torch.softmax(out, dim=1)[:,1].cpu().numpy()
            pred = out.argmax(1).cpu().numpy()

            preds.extend(pred)
            trues.extend(y.numpy())
            probs.extend(p)
    return np.array(preds), np.array(trues), np.array(probs)

# ==============================
# VISUALIZATION
# ==============================
def plot_cm(trues, preds, title, path):
    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.savefig(path)
    plt.close()

def plot_roc(trues, probs, path):
    fpr, tpr, _ = roc_curve(trues, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.legend()
    plt.savefig(path)
    plt.close()

def plot_tsne(features, labels, path):
    tsne = TSNE(n_components=2, random_state=SEED)
    X_emb = tsne.fit_transform(features)
    plt.figure()
    plt.scatter(X_emb[:,0], X_emb[:,1], c=labels, s=5)
    plt.savefig(path)
    plt.close()

# ==============================
# EXPERIMENTS
# ==============================
def run_experiment(train_X, train_y, test_X, test_y, name):
    model = CNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(AudioDataset(train_X, train_y), batch_size=32, shuffle=True)
    test_loader = DataLoader(AudioDataset(test_X, test_y), batch_size=32)

    for epoch in range(5):
        loss = train(model, train_loader, opt, loss_fn)
        preds, trues, probs = evaluate(model, test_loader)
        acc = (preds==trues).mean()
        print(f"{name} Epoch {epoch+1} | Loss {loss:.3f} | Acc {acc:.3f}")

    plot_cm(trues, preds, f"{name} CM", f"{OUT}/{name}_cm.png")
    plot_roc(trues, probs, f"{OUT}/{name}_roc.png")

    return model, preds, trues, probs

# ==============================
# MAIN
# ==============================
print("Loading datasets...")

# EXP 1 (In-domain)
X_s1, y_s1 = load_data(RAVDESS, 0)
X_s2, y_s2 = load_data(CREMA, 0)
X_l, y_l = load_data(LAUGH, 1)

X = X_s1 + X_s2 + X_l
y = y_s1 + y_s2 + y_l

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

model1, p1, t1, pr1 = run_experiment(X_tr, y_tr, X_te, y_te, "EXP1_in_domain")

# EXP 2 (Cross-corpus)
X_ehehe, y_ehehe = load_data(EHEHE, 1)
model2, p2, t2, pr2 = run_experiment(X, y, X_ehehe, y_ehehe, "EXP2_cross")

# EXP 3 (Adaptation)
Xe_tr, Xe_te, ye_tr, ye_te = load_ehehe_split()

X_adapt = X + Xe_tr
y_adapt = y + ye_tr

model3, p3, t3, pr3 = run_experiment(X_adapt, y_adapt, Xe_te, ye_te, "EXP3_adapt")

# ==============================
# t-SNE (final)
# ==============================
features = np.array([f.flatten() for f in X[:2000]])
labels = np.array(y[:2000])
plot_tsne(features, labels, f"{OUT}/tsne.png")

print("\nALL FIGURES SAVED IN:", OUT)