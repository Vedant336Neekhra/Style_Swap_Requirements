# importing necessay libraries
import torch
import torchvision.transforms as TF
import numpy as np
import clip

clip_model, preprocess = clip.load('RN50x4')
clip_model.eval()

def CLIP_encode_text(text_list:list) -> torch.Tensor:
    # Encodes given list of tokens and return the text embeddings in the shape of (batch_size, 512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    txt_tokens = clip.tokenize(text_list).to(device)
    txt_features = clip_model.encode_text(txt_tokens).float()
    txt_features_ = txt_features/txt_features.norm(dim=-1, keepdim=True)
    return txt_features_

def CLIP_encode_image(image_list, raw_image:bool=False):
    # Encodes given list of images (or tensor) and return the image embeddings in the shape of (batch_size, 512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # preprocess if rawimage
    if raw_image:
        img_tensors = []
        for img in image_list:
            img_tensors.append(preprocess(img))
        img_input = torch.Tensor(np.stack(img_tensors)).to(device)
    else:
        transform = TF.Compose([
            TF.Resize(size=224, interpolation=TF.InterpolationMode.BICUBIC, max_size=None, antialias=True)
            TF.CenterCrop(size=(224,224)),
            TF.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        img_input = transform(image_list).to(device)
        # img_input = image_list.to(device)
    img_features = clip_model.encode_image(img_input).float()
    img_features_ = img_features/img_features.norm(dim=-1, keepdim=True)
    return img_features_

def CLIP_loss(text_list, image_list, raw_image:bool=False):
    txt_embed = CLIP_encode_text(text_list)
    img_embed = CLIP_encode_image(image_list, raw_image)
    similarity = img_embed @ txt_embed.T
    return (1 - similarity).sum()
