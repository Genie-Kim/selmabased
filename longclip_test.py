from models.LongCLIP import longclip
import torch
from PIL import Image
from transformers import CLIPTokenizer
from diffusers import StableDiffusionPipeline,  UNet2DConditionModel, StableDiffusionXLPipeline, DiffusionPipeline


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = longclip.load("pretrained_models/LongCLIP/longclip-B.pt", device=device)

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16, cache_dir="/home/compu/JinProjects/jinprojects/SELMA/pretrained_models")
pipe.to(device)

text =["A man is crossing the street with a red car parked nearby.", "A man is driving a car in an urban scene."]


clip_tokenized = pipe.tokenizer(
    text,
    padding="max_length",
    max_length=248,
    truncation=True,
    return_tensors="pt",
).input_ids.to(device)


longclip_tokenized = longclip.tokenize(text).to(device)

image = preprocess(Image.open("models/LongCLIP/demo.png")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    longcliptokenizer_text_features = model.encode_text(longclip_tokenized)
    cliptokenizer_text_features = model.encode_text(clip_tokenized)
    
    logits_per_image = image_features @ longcliptokenizer_text_features.T
    longclip_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
    logits_per_image = image_features @ cliptokenizer_text_features.T
    clip_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", longclip_probs) 
print("Label probs:", clip_probs) 
# Label probs: [[0.9365 0.0637]]

