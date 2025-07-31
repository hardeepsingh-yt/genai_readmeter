# works using GenAI model, gives correct result

import os
import re
from PIL import Image
import logging
import sys
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log Python executable, version, and library versions
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Python version: {sys.version}")
try:
    import transformers
    logger.info(f"Transformers version: {transformers.__version__}")
    logger.info(f"Torch version: {torch.__version__}")
except ImportError as e:
    logger.error(f"ImportError: {e}")
    sys.exit(1)

# Check for required dependencies
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    logger.info("Imported torch, transformers, and qwen_vl_utils successfully")
except ImportError as e:
    logger.error(f"ImportError: {e}")
    logger.error("Required libraries are missing. Please install them using:")
    logger.error("python -m pip install --upgrade torch transformers pillow qwen-vl-utils accelerate")
    logger.error("For transformers, try: python -m pip install git+https://github.com/huggingface/transformers")
    sys.exit(1)

# Define the folder containing the images
image_folder = "test"
output_folder = "test_done"

# Verify input folder exists
image_folder = os.path.abspath(image_folder)
if not os.path.exists(image_folder):
    logger.error(f"Input folder '{image_folder}' does not exist.")
    sys.exit(1)

# Create output folder if it doesn't exist
output_folder = os.path.abspath(output_folder)
os.makedirs(output_folder, exist_ok=True)

# Load the Qwen2-VL-2B-Instruct model and processor
model_name = "Qwen/Qwen2-VL-2B-Instruct"
hf_token = os.getenv("HF_TOKEN")  # Set HF_TOKEN environment variable or add token here
try:
    logger.info(f"Loading model {model_name}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cpu", token=hf_token, force_download=True,
        ignore_mismatched_sizes=True  # Attempt to ignore size mismatches
    )
    processor = AutoProcessor.from_pretrained(model_name, token=hf_token, force_download=True)
    logger.info("Model and processor loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.error("Ensure you have an active internet connection and ~5GB free disk space.")
    logger.error("Try clearing cache: rmdir /s /q %USERPROFILE%\\.cache\\huggingface\\hub")
    logger.error("If the model is private, set HF_TOKEN environment variable or pass token directly.")
    logger.error("Try updating transformers: python -m pip install git+https://github.com/huggingface/transformers")
    sys.exit(1)

# Function to extract numbers from an image
def extract_number_from_image(image_path):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        prompt = "Extract the numerical reading from the meter in the image. Return only the number (no units or other text)."
        messages = [
            {"role": "system", "content": "You are a helper specialized in extracting numerical readings from meter images."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        # Prepare inputs for the model
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], 
            images=[image], 
            return_tensors="pt"
        ).to("cpu")
        logger.info(f"Processing {image_path} on CPU")
        
        # Generate the response
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract numbers (including decimals) from the response
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            logger.info(f"Extracted number: {numbers[0]} from {image_path}")
            return numbers[0]
        else:
            logger.warning(f"No number found in response: {response}")
            return None
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None

# Process all images in the folder
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(image_folder, filename)
        logger.info(f"Processing image: {filename}")
        
        # Extract number from the image
        number = extract_number_from_image(image_path)
        
        if number:
            # Sanitize the number to create a valid filename
            sanitized_number = number.replace('.', '_')
            new_filename = f"{sanitized_number}.jpg"
            new_filepath = os.path.join(output_folder, new_filename)
            
            # Check if the filename already exists and append a counter if necessary
            counter = 1
            while os.path.exists(new_filepath):
                new_filename = f"{sanitized_number}_{counter}.jpg"
                new_filepath = os.path.join(output_folder, new_filename)
                counter += 1
            
            # Copy the image to the new location with the new name
            try:
                Image.open(image_path).save(new_filepath, "JPEG", quality=95)
                logger.info(f"Renamed {filename} to {new_filename}")
            except Exception as e:
                logger.error(f"Error saving image {filename} as {new_filename}: {e}")
        else:
            logger.warning(f"Skipping {filename}: No number detected")

logger.info("Processing complete.")