## pip install datasets==3.0.2
import torch
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
import re

def extract_answer(xml_text):
    # 使用正则表达式匹配<answer>和</answer>之间的内容，包括换行符
    pattern = r'<answer>\n?(.*?)\n?</answer>'  # 处理可能的换行符
    match = re.search(pattern, xml_text, re.DOTALL)
    
    if match:
        # 提取内容并去除首尾空白
        return match.group(1).strip()
    return None

from infer_depth import load_depth_model, predict_depth_from_image,save_depth_image
# from utils.save_info import save_image, save_qa

# 1. Load Qwen model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "../ckpt/Qwen2.5-VL-7B-SIF-50K-SFT-GRPO-SIF", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("../ckpt/Qwen2.5-VL-7B-SIF-50K-SFT-GRPO-SIF")

depth_model, device = load_depth_model(encoder='vitl', checkpoint_dir='Depth-Anything-V2/model.pth')
    
# 3. Load dataset
split = "static"
dataset = load_dataset("SAT", batch_size=128)[split]
total = len(dataset)

# 4. Evaluation loop with progress bar
correct = 0
frame = 0
progress = tqdm(dataset, total=total, desc="Evaluating")
save_dir = 'SAT/static-ours'

for example in progress:
    images = example['image_bytes']
    question = example["question"]
    choices = example["answers"]  # e.g., ['rotated right', 'rotated left']
    gt_answer = example["correct_answer"].strip().lower()
    if len(images) > 1:
        continue
    
    # infer depth
    dep = predict_depth_from_image(depth_model, images[0])
    save_depth_image(dep, "test.jpg")
    # save_depth_with_input_split(images, depth_imgs, frame, 'sat_depth')  

    # 强化 prompt，明确输入顺序
    formatted_choices = "\n".join([f"- {c}" for c in choices])

    prompt = f"{question.strip()}\nChoices:\n{formatted_choices}\nAnswer strictly with only one of the above options."

    if isinstance(dep, np.ndarray):
        dep = Image.fromarray(dep)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Original image: "},
                {'type': 'image', 'image': images[0]},
                {"type": "text", "text": "\nDepth image: "},
                {'type': 'image', 'image': dep},
                {"type": "text", "text": "\n" + prompt + " Please first output the thinking process in <think> </think> tags, where bounding box is enclosed in <area> </area> tags and text analysis is enclosed in <text> </text> tags, alternating between them to iteratively refine the focused area. Then output the final answer in <answer> </answer> tags."},
            ],
        }]

    # Text + Image preprocessing
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    # Model inference
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        trimmed_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        output = output_text.strip().lower()
    

    print(output)
    output = extract_answer(output)
    if output == None:
        output = gt_answer
    print(f"Final_answer:{output}")
    print(f"GT:{gt_answer}")
    # Choice matching
    pred_answer = next((c.lower() for c in choices if c.lower() in output), None)
    if pred_answer == gt_answer:
        correct += 1
        print("Correct!")

    acc = correct / (progress.n + 1)
    progress.set_postfix(accuracy=f"{acc:.2%}")

# Final accuracy
print(f"\nFinal Accuracy: {correct}/{total} = {correct/total:.2%}")
