import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import csv

# https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in-pytorch-313f06910379
# https://wonder-j.tistory.com/11

torch.manual_seed(100)

if torch.cuda.is_available():
    print("Use GPU")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCH = 200

midi_features = []
emotions = []

def tempo_bin(tempo):
    if tempo > 1200000:    # bpm < 50
        return 0
    elif tempo > 1000000:  # bpm < 60
        return 1
    elif tempo > 800000:   # bpm < 75
        return 2
    elif tempo > 600000:   # bpm < 100
        return 3
    elif tempo >= 444445:  # bpm < 135
        return 4
    elif tempo >= 333334:  # bpm < 180
        return 5
    else:                  # bpm > 180
        return 6
        
def valence_category(valence):
    if valence < 0.0:
        return "L"
    elif valence < 0.5290476:
        return "M"
    else:
        return "H"

def arousal_category(arousal):
    if arousal < -0.1404167:
        return "L"
    elif arousal < 0.3381111:
        return "M"
    else:
        return "H"

def category_to_index(category):
    if category == 'LL': return 0
    elif category == 'LM': return 1
    elif category == 'LH': return 2
    elif category == 'ML': return 3
    elif category == 'MM': return 4
    elif category == 'MH': return 5
    elif category == 'HL': return 6
    elif category == 'HM': return 7
    elif category == 'HH': return 8
    else: return -1

with open('./output/extracted/vgmidi_emotion.csv', 'r') as raw:
    reader = csv.DictReader(raw)
    for line in reader:
        if int(line["ending"]) == 1: continue
        entity = [int(line["key.local.major"]), int(line["key.global.major"]),
            float(line["chord.maj"]) / 16, float(line["chord.min"]) / 16,
            float(line["chord.aug"]) / 16, float(line["chord.dim"]) / 16,
            float(line["chord.sus4"]) / 16, float(line["chord.dom7"]) / 16,
            float(line["chord.min7"]) / 16, float(line["note.density"]) / 16,
            float(line["note.pitch.mean"]) / 127, float(line["note.velocity"]) / 127,
            float(line["rhythm.density"]) / 16]
        entity.extend(np.eye(7)[tempo_bin(float(line["tempo"]))])
        entity.extend(np.eye(85)[int(line["roman.numeral"])])
        midi_features.append(entity)

        emotion_entity = [(float(line["valence"]) + 1) / 2, (float(line["arousal"]) + 1) / 2]
        emotions.append(emotion_entity)
    
midi_features = torch.from_numpy(np.array(midi_features, dtype=np.float32))
emotions = torch.from_numpy(np.array(emotions, dtype=np.float32))
print(midi_features.shape)
print(emotions.shape)

x = Variable(midi_features)
y = Variable(emotions)

torch_dataset = Data.TensorDataset(x, y)

train_set, val_set = Data.random_split(torch_dataset, [9395, 1000])

train_loader = Data.DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = Data.DataLoader(dataset=val_set)

net = torch.nn.Sequential(
    torch.nn.Linear(105, 256),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(64, 2)
).to(device)

optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()

for epoch in range(EPOCH):

    running_loss = 0.0
    for step, (batch_x, batch_y) in enumerate(train_loader):
        b_x = Variable(batch_x).to(device)
        b_y = Variable(batch_y).to(device)

        prediction = net(b_x)
        loss = loss_func(prediction, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    print('[%d] loss: %.6f' % (epoch + 1, running_loss / BATCH_SIZE))

print('Training finished!')

torch.save(net.state_dict(), './regression_model.pth')

"""
# load trained network parameters
net = Net()
net.load_state_dict(torch.load('./regression_model.pth'))
"""

with torch.no_grad():
    val_loss = 0.0
    category_accuracy = 0.0
    category_matrix = np.zeros((9, 9), dtype=int)
    for data in val_loader:
        val_x, val_y = data[0].to(device), data[1].to(device)

        prediction = net(val_x)
        loss = loss_func(prediction, val_y)

        gt_category = valence_category(val_y[0, 0].item() * 2 - 1) + arousal_category(val_y[0, 1].item() * 2 - 1)
        pred_category = valence_category(prediction[0, 0].item() * 2 - 1) + arousal_category(prediction[0, 1].item() * 2 - 1)
        
        val_loss += loss.item()
        category_matrix[category_to_index(gt_category), category_to_index(pred_category)] += 1
        if gt_category == pred_category:
            category_accuracy += 1

        print(val_y, gt_category, " | ", prediction, pred_category, gt_category == pred_category)

    print('training loss: %.6f' % (running_loss / BATCH_SIZE))
    print('validation loss: %.6f' % (val_loss / 1000))
    print('category_accuracy: %.3f' % (category_accuracy / 1000))
    print(category_matrix)