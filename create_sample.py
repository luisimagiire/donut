import io
import json

import datasets
from PIL import Image
from tqdm import tqdm
import os


def main():
    # Read parquet files in cord-v2 folder
    df = datasets.load_dataset("naver-clova-ix/cord-v2")
    sets = ['validation', 'train', 'test']

    for s in tqdm(sets):
        _counter = 0
        for img, label in zip(df.data[s]['image'], df.data[s]['ground_truth']):
            im = Image.open(io.BytesIO(img[0].as_buffer()))
            save_path = f'/home/lhmoi/pojetos/ml_lib/donut/cord_dataset_sample/{s}/img__{_counter}.jpg'
            im.save(save_path)
            with open(f'/home/lhmoi/pojetos/ml_lib/donut/cord_dataset_sample/{s}/metadata.jsonl', 'a+') as f:
                _jsonl = {'file_name': os.path.basename(save_path), 'ground_truth': json.dumps(json.loads(label.as_py()))}
                f.write(json.dumps(_jsonl) + "\n")
            _counter += 1


if __name__ == '__main__':
    main()
