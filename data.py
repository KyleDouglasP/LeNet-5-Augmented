from datasets import load_dataset
from PIL import Image
import os

# Use same function signature and names as original
def write_images(dataset, split):
    image_dir = f"./data/{split}/"
    label_file = f"./data/{split}_label.txt"
    os.makedirs(image_dir, exist_ok=True)

    with open(label_file, "w") as label_f:
        for i, example in enumerate(dataset):
            image = example['image']
            label = example['label']
            image.save(f"{image_dir}{i}.png")
            label_f.write(f"{label}\n")

# Use Hugging Face dataset loading, but keep the same variable names
splits = {
    "train": "mnist/train-00000-of-00001.parquet",
    "test": "mnist/test-00000-of-00001.parquet"
}
dataset = load_dataset("ylecun/mnist")

df_train = dataset["train"]
df_test = dataset["test"]

write_images(df_train, "train")
write_images(df_test, "test")