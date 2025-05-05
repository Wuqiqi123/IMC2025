from cust3r.inference import inference, inference_recurrent
from cust3r.model import ARCroco3DStereo
cust3r_model_name = "/workspace/IMC2025/ckpts/cut3r_224_linear_4.pth"
model = ARCroco3DStereo.from_pretrained(cust3r_model_name)
model.eval()