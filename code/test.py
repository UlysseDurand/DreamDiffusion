from EEGPT import LitEEGPTCausal
import torch
import numpy as np

model = LitEEGPTCausal(load_path="eegpt_mcae_58chs_4s_large4E.ckpt")
print(model.encode(torch.from_numpy(np.zeros((200, 19, 512))).to(torch.float32)).shape)