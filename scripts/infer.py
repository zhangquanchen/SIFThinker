import os
import re
import base64
from PIL import Image, ImageDraw
import os
from openai import OpenAI
import json

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

client = OpenAI(
    api_key="{}".format(os.getenv("API_KEY", "0")),
    base_url="http://localhost:{}/v1".format(os.getenv("API_PORT", 8020)),
)

def extract_all_bounding_boxes(content):
    result = []
    for area_str in re.findall(r"<area>(.*?)</area>", content, re.DOTALL):
        boxes = []
        try:
            area_json = json.loads(area_str)
            
            if isinstance(area_json, list):
                for item in area_json:
                    if "bbox_2d" in item:
                        coords = item["bbox_2d"]
                        boxes.append(coords)
            else:
                if "bbox_2d" in area_json:
                    coords = area_json["bbox_2d"]
                    boxes.append(coords)
        except Exception as e:
            print(f"Analisys <area> error: {e}")
            boxes.append([])
        result.append(boxes)
    return result

def visualize_bboxes(image_path, bounding_boxes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.basename(image_path)
    base_name, ext = os.path.splitext(image_name)
    
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        for i, box in enumerate(bounding_boxes, start=1):
            img_copy = img.copy()
            draw = ImageDraw.Draw(img_copy)
            
            for bbx in box:
                x1 = int(bbx[0] * width)
                y1 = int(bbx[1] * height)
                x2 = int(bbx[2] * width)
                y2 = int(bbx[3] * height)
                
                draw.rectangle([x1, y1, x2, y2], outline='red', width=10)
            
            output_path = os.path.join(output_dir, f"{base_name}_{i}{ext}")
            img_copy.save(output_path)
            print(f"Save to: {output_path}")
            
    except Exception as e:
        print(f"Handle image error: {e}")

if __name__ == "__main__":
    subfix = " Please first output the thinking process in <think> </think> tags, where bounding box is enclosed in <area> </area> tags and text analysis is enclosed in <text> </text> tags, alternating between them to iteratively refine the focused area. Then output the final answer in <answer> </answer> tags."
    prompt = "What is the color of the dog?"
    image_path = "../data/vstar_bench/direct_attributes/sa_717.jpg"
    output_dir = "output_bbox"
    base64_image = encode_image(image_path)
    base_dir = os.path.dirname(image_path)
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    image_depth_path = os.path.join(base_dir, f"{file_name}_d.png")
    base64_image_depth = encode_image(image_depth_path)
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Original image: "},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                {"type": "text", "text": "\nDepth image: "},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_depth}"}},
                {"type": "text", "text": "\n" + prompt + subfix},
            ],
        }
    ]

    response = client.chat.completions.create(messages=messages, model="test",max_tokens=2048)
    text_str = response.choices[0].message.content
    print(text_str)
    bounding_boxes = extract_all_bounding_boxes(text_str)
    print(f"\nFind {len(bounding_boxes)} bounding boxes")
    
    visualize_bboxes(image_path, bounding_boxes, output_dir)