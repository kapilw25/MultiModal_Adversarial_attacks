# pip install albumentations opencv-python pillow adversarial-robustness-toolbox


import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import albumentations as A

# === 1. Define semantic obfuscations (PIL) ===
def overlay_indicator(img_pil, text="!!! MALWARE !!!", opacity=128):
    txt = Image.new("RGBA", img_pil.size, (255,255,255,0))
    draw = ImageDraw.Draw(txt)
    font = ImageFont.load_default()
    w, h = draw.textsize(text, font=font)
    pos = (img_pil.width - w - 10, 10)
    draw.text(pos, text, fill=(255,0,0,opacity), font=font)
    return Image.alpha_composite(img_pil.convert("RGBA"), txt)

def inject_log_noise(img_pil, lines=3):
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default()
    for i in range(lines):
        y = img_pil.height - (i+1)*(font.getsize("A")[1]+2)
        noise = f"10.{np.random.randint(0,255)}.{np.random.randint(0,255)} INFO: UA Login"
        draw.text((5, y), noise, fill=(0,0,0), font=font)
    return img_pil

def watermark_field(img_pil, mark="CONFIDENTIAL", opacity=64):
    txt = Image.new("RGBA", img_pil.size, (255,255,255,0))
    draw = ImageDraw.Draw(txt)
    font = ImageFont.load_default()
    w, h = draw.textsize(mark, font=font)
    # center
    pos = ((img_pil.width - w)//2, (img_pil.height - h)//2)
    draw.text(pos, mark, fill=(0,0,0,opacity), font=font)
    return Image.alpha_composite(img_pil.convert("RGBA"), txt)

# === 2. Black-box optical distortions (Albumentations) ===
bb_augment = A.Compose([
    A.OpticalDistortion(distort_limit=0.3, shift_limit=0.05, p=1.0),
    A.Perspective(scale=(0.05, 0.1), p=1.0),
    A.MotionBlur(blur_limit=5, p=0.7),
])

# === 3. (Optional) White-box PGD attack via ART ===
try:
    from art.attacks.evasion import ProjectedGradientDescent
    from art.estimators.classification import PyTorchClassifier
    import torch

    # placeholder: replace with your actual model + preprocessing
    def make_classifier(model, device):
        model.to(device).eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        # define input shape, clip min/max
        clip_classifier = PyTorchClassifier(
            model=model,
            clip_values=(0.0,1.0),
            loss=loss_fn,
            input_shape=(3,224,224),
            nb_classes=1000,
            device_type='gpu'
        )
        return clip_classifier

    def pgd_attack(classifier, np_imgs, eps=2/255, eps_step=0.5/255, max_iter=10):
        attack = ProjectedGradientDescent(
            estimator=classifier,
            norm=np.inf,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter
        )
        return attack.generate(np_imgs)
except ImportError:
    pgd_attack = None

# === 4. Pipeline over benchmark images ===
input_dir = "benchmark_images/"
output_dir = "benchmark_images_obf/"
os.makedirs(output_dir, exist_ok=True)

# If you want to do white-box, prepare your classifier:
# device = torch.device("cuda:0")
# my_classifier = make_classifier(my_clip_model, device)

for fname in os.listdir(input_dir):
    if not fname.lower().endswith((".png",".jpg")): continue
    path = os.path.join(input_dir, fname)
    img_cv2 = cv2.imread(path)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

    # 1) Semantic obfuscations
    img_pil = overlay_indicator(img_pil)
    img_pil = inject_log_noise(img_pil)
    img_pil = watermark_field(img_pil)

    # convert back to array
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGR)

    # 2) Black-box distortions
    img = bb_augment(image=img)["image"]

    # 3) White-box PGD (optional)
    if pgd_attack:
        # normalize [0,1], shape(N,H,W,C)
        x = np.expand_dims(img.astype("float32")/255.0, axis=0)
        x_adv = pgd_attack(my_classifier, x)
        img = (x_adv[0] * 255).astype("uint8")

    # save
    cv2.imwrite(os.path.join(output_dir, fname), img)

print("All done — you now have semantically-obfuscated, distorted (and even PGD-attacked) images ready for inference testing.")
