
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys

sys.path.append(".")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import score
from utils.dataset import imagenet_loaders
from utils.metrics import SoftmaxClassification, OOD
from torchvision import models
args = manage_experiment_configuration()

torch.manual_seed(args.seed)

train_loader, test_loader = imagenet_loaders(args, valid_size=0.0)

import torchvision.models as models

f = models.__dict__[args.resnet](pretrained=True).to(args.device).to(args.dtype)
f.eval()

save_str = "MAP_ImageNet_dataset={}".format(
    args.dataset_name)


def test_step(X, y):

        # In case targets are one-dimensional and flattened, add a final dimension.
    if y.ndim == 1:
        y = y.unsqueeze(-1)

        # Cast types if needed.
    if args.dtype != X.dtype:
        X = X.to(args.dtype)
    if args.dtype != y.dtype:
        y = y.to(args.dtype)

    Fmean = f(X)  # Forward pass
    Fvar = torch.zeros(Fmean.shape[0], Fmean.shape[1], Fmean.shape[1],
                       dtype = args.dtype, device = args.device)

    return 0, Fmean, Fvar

f.test_step = test_step


test_metrics = score(
    f,
    test_loader,
    SoftmaxClassification,
    use_tqdm=True,
    device=args.device,
    dtype=args.dtype,
)

print(test_metrics)
input()


test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations
test_metrics["seed"] = args.seed
test_metrics["time"] = 0


df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()

print(df)

df.to_csv(
    path_or_buf="results/" + save_str + ".csv",
    encoding="utf-8",
)
