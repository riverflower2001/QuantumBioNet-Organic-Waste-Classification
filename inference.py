"""
Quantum BioNet 2.0
This repository accompanies the published paper and is provided for
academic inspection and educational purposes.

NOTE:
- Google Colab-specific code has been removed.
- Dataset paths and pretrained weights are intentionally omitted.
- Additional resources are required to reproduce the published results.
"""


# MOUNT GOOGLE DRIVE
import zipfile
import os
import shutil

zip_path = "YOUR_DATASET_PATH"
extract_to = "YOUR_DATASET_PATH"

if os.path.exists(extract_to):
    shutil.rmtree(extract_to)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Top-level folders/files after extraction:")
for item in os.listdir(extract_to):
    print(item)

# ---- Section ----

# 1️⃣3️⃣ Hierarchical Prediction on a Random Test Image
def test_random_image(base_dir, model_stage1, model_stage2, device, class_names_stage2, img_size=(224,224)):
    all_images = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))
    if not all_images:
        print("No images found."); return
    random_image = random.choice(all_images)
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])
    img = Image.open(random_image).convert('RGB')
    plt.imshow(img); plt.axis('off'); plt.title("Test Image"); plt.show()
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    model_stage1.eval(); model_stage2.eval()
    with torch.no_grad():
        out1 = torch.sigmoid(model_stage1(img_tensor))
        stage1_prob = out1.item()
        stage1_label = 'Organic' if stage1_prob > 0.5 else 'Inorganic'
        print(f"Stage 1: {stage1_label} (Conf: {stage1_prob:.2%})" if stage1_label == "Organic"
              else f"Stage 1: Inorganic (Conf: {(1-stage1_prob):.2%})")
        if stage1_label == 'Organic':
            out2 = torch.softmax(model_stage2(img_tensor), dim=1)
            class_idx = torch.argmax(out2, 1).item()
            stage2_label = class_names_stage2[class_idx]
            stage2_conf = out2[0, class_idx].item()
            print(f"Stage 2: {stage2_label} (Conf: {stage2_conf:.2%})")
        else:
            print("Stage 2: Needs special handling (Conf: 1.00)")

# EXAMPLE USAGE:
# test_random_image("YOUR_DATASET_PATH",
#                   model_stage1, model_stage2, device, class_names2, img_size=IMG_SIZE)

# ---- Section ----

test_random_image("YOUR_DATASET_PATH",
                   model_stage1, model_stage2, device, class_names2, img_size=IMG_SIZE)