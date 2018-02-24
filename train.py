import os
import sys
import json
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchnet.dataset import SplitDataset, ShuffleDataset
import torch.nn.functional as F
from math import sqrt
from onnx_coreml import convert
import onnx

number_of_points = 100
number_of_channels = 2
epochs = 1000
batch_size = 6
current_dir = os.path.dirname(__file__)
data_file = os.path.join(current_dir, 'data.json')


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=number_of_channels, out_channels=channels, kernel_size=(1, 7), padding=(0, 3)),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=(1, 3), padding=(0, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, x.size(1), 1, x.size(2))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, x.size(3))
        return x


def encode(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    y_shift = ((max_y - min_y) / (max_x - min_x)) / 2.0
    input_tensor = torch.zeros([number_of_channels, number_of_points])

    def normalize_x(x):
        return (x - min_x) / (max_x - min_x) - 0.5
    def normalize_y(y):
        return (y - min_y) / (max_x - min_x) - y_shift

    for i in range(min(number_of_points, len(points))):
        x = points[i][0] * 1.0
        y = points[i][1] * 1.0
        input_tensor[0][i] = normalize_x(x)
        input_tensor[1][i] = normalize_y(y)
        continue
    return input_tensor


class PointsDataset(Dataset):
    def __init__(self, csv_file):
        self.examples = json.load(open(csv_file))
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_tensor = encode(example['points'])
        output_tensor = torch.zeros(number_of_points)
        for split_position in example['splits']:
            index = next(i for i, point in enumerate(example['points']) if point[0] > split_position) - 1
            output_tensor[index] = 1
        return input_tensor, output_tensor

def evaluate(model, data):
    inputs, target = data
    inputs = Variable(inputs)
    target = Variable(target)
    mask = inputs.eq(0).sum(dim=1).eq(0)
    logits = model(inputs)
    correct = int(logits.round().eq(target).mul(mask).sum().data)
    total = int(mask.sum())
    accuracy = 100.0 * correct / total

    float_mask = mask.float()
    masked_logits = logits.mul(float_mask)
    masked_target = target.mul(float_mask)
    loss = F.binary_cross_entropy(masked_logits, masked_target)

    return float(loss), accuracy, correct, total

def train(model, epochs=epochs, batch_size=batch_size):
    optimizer = torch.optim.Adam(model.parameters())
    dataset = PointsDataset(data_file)
    dataset = SplitDataset(ShuffleDataset(dataset), {'train': 0.9, 'validation': 0.1})
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    model.train()

    for epoch in range(epochs):
        dataset.select('train')
        running_loss = 0.0

        for i, (inputs, target) in enumerate(loader):
            inputs = Variable(inputs)
            target = Variable(target)

            logits = model(inputs)
            mask = inputs.eq(0).sum(dim=1).eq(0)
            float_mask = mask.float()
            masked_logits = logits.mul(float_mask)
            masked_target = target.mul(float_mask)
            loss = F.binary_cross_entropy(masked_logits, masked_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

 
        dataset.select('validation')
        validation_loss, validation_accuracy, correct, total = evaluate(model, next(iter(loader)))

        print '\r[{:4d}] - running loss: {:8.6f} - validation loss: {:8.6f} validation acc: {:7.3f}% ({}/{})'.format(
                epoch + 1,
                running_loss,

                validation_loss,
                validation_accuracy,
                correct,
                total
            ),
        sys.stdout.flush()

        running_loss = 0.0

    print('\n')


if __name__ == '__main__':
    model = Model()
    train(model)
    path = os.path.join(current_dir, 'SplitModel.proto')
    dummy_input = Variable(torch.FloatTensor(1, number_of_channels, number_of_points))
    torch.save(model.state_dict(), os.path.join(current_dir, 'SplitModel.pt'))
    torch.onnx.export(model, dummy_input, path, verbose=True)
    model = onnx.load(os.path.join(os.path.dirname(__file__), 'SplitModel.proto'))
    coreml_model = convert(
        model,
        'classifier',
        image_input_names=['input'],
        image_output_names=['output'],
        class_labels=[i for i in range(number_of_points)],
    )
    coreml_model.save('SplitModel.mlmodel')
