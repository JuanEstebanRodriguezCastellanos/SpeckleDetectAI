import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import cv2
from PIL import ImageOps

class UNetResNet50(nn.Module):
    def __init__(self, out_classes=1):
        super(UNetResNet50, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, out_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)

# ------------------------
# Uso del modelo
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo entrenado
model = UNetResNet50()
model.load_state_dict(torch.load('modelo50.pth', map_location=device))
model = model.to(device)
model.eval()

# Transformación de imagen
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def pad_image(im, size=(512, 512), color=(0,)):
    # Mantener relación de aspecto
    ratio = min(size[0]/im.size[0], size[1]/im.size[1])
    new_size = (int(im.size[0]*ratio), (int(im.size[1]*ratio)))
    im = im.resize(new_size, Image.Resampling.LANCZOS)
    
    delta_w = size[0] - im.size[0]
    delta_h = size[1] - im.size[1]
    padding = (delta_w//2, delta_h//2, delta_w - delta_w//2, delta_h - delta_h//2)
    return ImageOps.expand(im, padding, fill=color), padding, im.size  # Devolver tamaño antes de padding

def procesar(image_pil):
    image_pil = image_pil.convert("RGB")
    original_size = image_pil.size
    
    # Padding manteniendo relación de aspecto
    image_padded, padding, padded_size = pad_image(image_pil, size=(512, 512), color=(0, 0, 0))
    
    # Redimensionar a 256x256 para el modelo
    image_resized = image_padded.resize((256, 256))
    img_tensor = transform(image_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    mask = output.squeeze().cpu().numpy()
    mask_binaria = (mask > 0.5).astype(np.uint8)

    # Redimensionar máscara al tamaño con padding (512x512)
    mask_padded = cv2.resize(mask_binaria, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Recortar el padding
    left, top, right, bottom = padding
    mask_cropped = mask_padded[top:top+padded_size[1], left:left+padded_size[0]]
    
    # Redimensionar al tamaño original
    mask_restored = cv2.resize(mask_cropped, original_size, interpolation=cv2.INTER_NEAREST)

    # Convertir imagen original a formato OpenCV
    imagen_cv = np.array(image_pil)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask_restored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dibujar contornos
    imagen_con_contornos = imagen_cv.copy()
    cv2.drawContours(imagen_con_contornos, contours, -1, (0, 255, 0), thickness=2)

    return Image.fromarray(imagen_con_contornos)