import json
from ultralytics.nn.text_model import MobileCLIP

mobileclip = MobileCLIP("blt", "cpu")

with open("/Users/chnam/Projects/Research/yoloe/converter/final_mixed_train_no_coco_segm_korean.json", "r") as f:
    korean_data = json.load(f)


texts = [item["caption"] for item in korean_data["images"]]

text_features = mobileclip.tokenize(texts)

print(text_features)