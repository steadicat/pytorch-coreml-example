#!/usr/bin/python
import json
import os
import torch
from torch.autograd import Variable
from train import Model, encode

current_dir = os.path.dirname(__file__)
model_file = os.path.join(current_dir, 'SplitModel.pt')
data_file = os.path.join(current_dir, 'data.json')
output_file = os.path.join(current_dir, 'prediction.json')

model = Model()
model.load_state_dict(torch.load(model_file))

with open(data_file) as f:
    examples = json.load(f)

data = Variable(torch.stack([encode(p['points']) for p in examples]))
logits = model(data)
prediction = []
for example, probs in zip(examples, logits):
    prediction.append([i for i, prob in enumerate(list(probs)) if float(prob) >= 0.5 and i < len(example['points'])])

with open(output_file, 'w') as f:
    json.dump(prediction, f)
