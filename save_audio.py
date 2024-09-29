import os
import json
import soundfile as sf
from datasets import load_dataset
import random

# 要下载的语言对列表
language_pairs = ["de_en", "zh-CN_en", "fr_en", "it_en", "es_en"]

# 各源语言对应的“翻译成英语”提示
translation_prompts = {
    "de": "Ins Englische übersetzen",        # 德语
    "zh-CN": "翻译成英语",                     # 中文
    "fr": "Traduire en anglais",             # 法语
    "it": "Traduci in inglese",              # 意大利语
    "es": "Traducir al inglés"               # 西班牙语
}

# 要下载的数据集划分
splits = ["train", "validation", "test"]

# 数据保存的基本目录
save_base_dir = "/root/autodl-tmp/audio/covost2"

# 初始化一个字典，用于收集每个划分的对话
split_conversations = {
    "train": [],
    "validation": [],
    "test": []
}

for lang_pair in language_pairs:
    source_lang, _ = lang_pair.split('_')
    # 获取源语言的翻译提示
    translation_prompt = translation_prompts[source_lang]

    for split in splits:
        # 加载数据集
        dataset = load_dataset("fixie-ai/covost2", lang_pair, split=split)
        # 创建保存目录
        split_dir = os.path.join(save_base_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for example in dataset:
            # 获取音频数据
            audio_data = example['audio']['array']
            sample_rate = example['audio']['sampling_rate']
            audio_file_name = example['id'] + ".wav"  # 添加文件扩展名
            audio_file_path = os.path.join(split_dir, audio_file_name)

            # 保存音频文件
            sf.write(audio_file_path, audio_data, sample_rate)

            # 创建对话
            conversation = {
                "conversations": [
                    {"from": "user", "value": f"<audio>{audio_file_name}</audio>{translation_prompt}"},
                    {"from": "assistant", "value": example['translation']}
                ]
            }
            # 添加到相应的划分中
            split_conversations[split].append(conversation)

# 对每个划分的对话进行打乱并保存
for split in splits:
    random.shuffle(split_conversations[split])
    # 修改JSON文件名，使其反映数据集划分
    json_file_name = f"covost2_{split}.json"
    json_file_path = os.path.join(save_base_dir, split, json_file_name)
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(split_conversations[split], json_file, ensure_ascii=False, indent=4)

print("所有数据已保存。")
