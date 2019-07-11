from data import *
from model import *
from train import *
from sample import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AAE().to(device)
fit(model,train_data)
model.eval()
get_samples(model)

