import torch.nn as nn

def get_model(num_classes):
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16*222*222, num_classes)
    )
    return model
