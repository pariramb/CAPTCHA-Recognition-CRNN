from src.model import create_model
from src.inference import Inference
from src.utils import load_checkpoint

model = create_model()
load_checkpoint(model, None, "Checkpoint.pth")
inference = Inference(model)