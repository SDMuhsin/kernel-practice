# %%
import torch
import torch.nn.functional as F
from kata_setup import TODO, S, check, same, is_blank   # open it if curious

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=110)
