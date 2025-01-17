import os
import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from diffusers import StableDiffusionPipeline
import torch
from skimage.metrics import structural_similarity as ssim
from utils import utils_ade20k  # Ensure this utility is available
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

def dummy_safety(images, **kwargs):
    return images, [False] * len(images)



# Constants
DATASET_PATH = r'C:\Users\citru\Downloads\vicky72_7e81d671'
INDEX_FILE = 'ADE20K_2021_17_01\index_ade20k.pkl'

# Load ADE20K index
index_file_path = os.path.join(DATASET_PATH, INDEX_FILE)
with open(index_file_path, 'rb') as f:
    index_ade20k = pkl.load(f)

print(f"Dataset loaded with {len(index_ade20k['filename'])} images.")



# Choose an image by index
image_index = 16868
file_name = index_ade20k['filename'][image_index]
folder_name = index_ade20k['folder'][image_index]
full_file_name = os.path.join(DATASET_PATH, folder_name, file_name)
scene = index_ade20k['scene'][image_index]
print(f"Selected image: {file_name}")

# Load segmentation and parts masks
info = utils_ade20k.loadAde20K(full_file_name)
img = cv2.imread(info['img_name'])[:, :, ::-1]  # Convert BGR to RGB
seg = cv2.imread(info['segm_name'])[:, :, ::-1]
seg_mask = info['class_mask']

# Verify unique IDs in segmentation mask
print("Unique values in class_mask:", np.unique(seg_mask))

# Material mapping based on segmentation IDs
# Replace these IDs based on actual unique IDs in seg_mask
material_map = {
    64: {'glossiness': 0.9, 'reflectivity': 0.8},   # Example: car
    312: {'glossiness': 0.1, 'reflectivity': 0.3},  # Example: building
}

# Debug: Check if segmentation IDs in material_map exist in seg_mask
for seg_id in material_map.keys():
    if seg_id not in np.unique(seg_mask):
        print(f"Warning: Segment ID {seg_id} not found in segmentation mask.")

# Function to create material maps
def create_material_maps(seg_mask, material_map):
    glossiness_map = np.zeros(seg_mask.shape, dtype=np.float32)
    reflectivity_map = np.zeros(seg_mask.shape, dtype=np.float32)

    for seg_id, props in material_map.items():
        glossiness_map[seg_mask == seg_id] = props['glossiness']
        reflectivity_map[seg_mask == seg_id] = props['reflectivity']

    return glossiness_map, reflectivity_map

# Generate glossiness and reflectivity maps
glossiness_map, reflectivity_map = create_material_maps(seg_mask, material_map)

# Debug: Check non-zero values
print("Glossiness Map Non-Zero Count:", np.count_nonzero(glossiness_map))
print("Reflectivity Map Non-Zero Count:", np.count_nonzero(reflectivity_map))

# Visualize material maps
def visualize_material_maps(glossiness_map, reflectivity_map):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(glossiness_map, cmap='plasma')
    plt.title("Glossiness Map")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reflectivity_map, cmap='plasma')
    plt.title("Reflectivity Map")
    plt.axis('off')

    plt.show()

visualize_material_maps(glossiness_map, reflectivity_map)

# Stable Diffusion image generation
from diffusers import StableDiffusionPipeline
import torch

model_path = "CompVis/stable-diffusion-v1-4"  # Change to your model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(model_path).to(device)
pipe.safety_checker = dummy_safety
# Generate image
def generate_image(prompt, negative_prompt):
    return pipe(prompt, negative_prompt=negative_prompt, guidance_scale=7.5, height=512, width=512).images[0]

prompt = "A glossy red sports car on a reflective street, photorealistic"
negative_prompt = "blurry, low quality"
generated_image = generate_image(prompt, negative_prompt)

# Visualize generated image
plt.figure(figsize=(7, 7))
plt.imshow(generated_image)
plt.axis('off')
plt.title("Generated Image")
plt.show()

# Evaluate image quality using SSIM
def evaluate_image_quality(original_map, generated_image):
    # Convert the generated image to grayscale
    generated_gray = cv2.cvtColor(np.array(generated_image), cv2.COLOR_RGB2GRAY)

    # Resize generated image to match original_map dimensions
    generated_gray_resized = cv2.resize(generated_gray, (original_map.shape[1], original_map.shape[0]))

    # Compute SSIM
    score = ssim(original_map, generated_gray_resized, data_range=generated_gray_resized.max() - generated_gray_resized.min())
    print(f"SSIM Score: {score:.2f}")

evaluate_image_quality(glossiness_map, generated_image)

# --- Main Pipeline ---
if __name__ == "__main__":
    # Step 1: Load Dataset
    index_ade20k = load_dataset_index(DATASET_PATH, INDEX_FILE)
    print(f"Dataset loaded with {len(index_ade20k['filename'])} images.")

    material_map = {
    401: {'glossiness': 0.9, 'reflectivity': 0.8},
    310: {'glossiness': 0.1, 'reflectivity': 0.3}
    }
    object_name_to_id = {name: i for i, name in enumerate(index_ade20k['objectnames'])}


    # Step 2: Load Specific Image Info
    image_index = 16868
    full_file_name, file_name, scene, num_objects, num_parts = load_image_info(index_ade20k, DATASET_PATH, image_index)

    print(f"Selected image: {file_name}")
    print(f"Scene: {scene}, Objects: {num_objects}, Parts: {num_parts}")

    # Step 3: Load Image Segmentation
    info = utils_ade20k.loadAde20K(full_file_name)
    img, seg_mask = visualize_image_and_segmentation(info, obj_id=400)

    # Step 4: Extract Texture Features
    texture_map = extract_texture(seg_mask)
    visualize_texture_map(texture_map)

    # Step 5: Generate Material Maps
    material_map = {
        'car': {'glossiness': 0.9, 'reflectivity': 0.8},
        'building': {'glossiness': 0.1, 'reflectivity': 0.3},
        # Add other mappings
    }
    glossiness_map, reflectivity_map = create_material_maps(seg_mask, info, material_map)
    visualize_material_maps(glossiness_map, reflectivity_map)

    # Step 6: Generate Image
    prompt = "A glossy red sports car on a reflective street, photorealistic"
    negative_prompt = "blurry, low quality"
    generated_image = generate_image(prompt, negative_prompt)

    # Step 7: Evaluate Image Quality
    evaluate_image_quality(glossiness_map, generated_image)
