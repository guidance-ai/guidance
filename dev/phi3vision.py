from guidance import models, gen, select, image, user, assistant, system
from guidance._grammar import string
PHI_3_VISION_MODEL = "microsoft/Phi-3-vision-128k-instruct"

model_kwargs = {
    "_attn_implementation": "eager", # Uncomment this line if flash attention is not working
    "trust_remote_code": True,
    "device_map": "mps",
}
phi3v = models.TransformersPhi3Vision(
    model=PHI_3_VISION_MODEL, **model_kwargs
)

lm = phi3v

with user():
    image_url = "https://picsum.photos/200/300"
    lm += "What do you see in this image?" + image(image_url)
    # lm += "What do you see in this image?"

with assistant():
    lm += gen(temperature=0.8)