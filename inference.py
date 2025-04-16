import os
import json
import random
import re
import ast
import subprocess
import string
import torch.multiprocessing as mp
from collections import defaultdict
from difflib import SequenceMatcher
from tqdm import tqdm
import argparse

try:
    from video_bench.models.qwen2vl import Qwen2VL 
except ImportError:
    print("failed to load qwen2vl")
try:
    from video_bench.models.mammoth_vl import MAmmoTH_VL 
except ImportError:
    print("failed to load mammoth_vl")
try:
    from video_bench.models.llava_video_image import LLaVA_Video
except ImportError:
    print("failed to load llava_video_image")
try:
    from video_bench.models.llamavid_7b import LLamaVID 
except ImportError:
    print("failed to load llamavid")
try:
    from video_bench.models.llavamini_8b import LLaVAMini 
except ImportError:
    print("failed to load llavamini")
try:
    from video_bench.models.longva_7b import LongVA 
except ImportError:
    print("failed to load longva")
try:
    from video_bench.models.longvila import LongVILA 
except ImportError:
    print("failed to load longvila")
try:
    from video_bench.models.nvila import NVILA 
except ImportError:
    print("failed to load nvila")
try:
    from video_bench.models.longvu_7b import LongVU 
except ImportError:
    print("failed to load longvu")
try:
    from video_bench.models.mplug_owl3 import mPluGOWL3 
except ImportError:
    print("failed to load mplug_owl3")
try:
    from video_bench.models.internvl2_5 import Internvl2_5
except ImportError:
    print("failed to load internvl2_5")
try:
    from video_bench.models.internvl2_5_lmdeploy import Internvl2_5
except ImportError:
    print("failed to load internvl2_5_lmdeploy")
try:
    from video_bench.models.internvl2 import Internvl2
except ImportError:
    print("failed to load internvl2")
try:
    from video_bench.models.llava_ov import LLaVA
except ImportError:
    print("failed to load llava")
try:
    from video_bench.models.llava_video import LLaVA_Video
except ImportError:
    print("failed to load llava_video")
try:
    from video_bench.models.phi3_5 import Phi3_5
except ImportError:
    print("failed to load phi3_5")
try:
    from video_bench.models.phi4 import Phi4
except ImportError:
    print("failed to load phi4")
try:
    from video_bench.models.llava_next_video import LLaVA_NV
except ImportError:
    print("failed to load LLaVA_NV")
try:
    from video_bench.models.minicpmv import MiniCPMV
except ImportError:
    print("failed to load minicpmv")
try:
    from video_bench.models.minicpmo import MiniCPMO
except ImportError:
    print("failed to load minicpmo")

from video_bench.models.aria import Aria
try:
    from video_bench.models.aria import Aria
except ImportError:
    print("failed to load aria")
try:
    from video_bench.models.qwen2_5vl import Qwen2_5VL
except ImportError:
    print("failed to load qwen2_5vl")
try:
    from video_bench.models.videollama3 import VideoLlama3
except ImportError:
    print("failed to load videollama3")
try:
    from video_bench.models.internvideo2_5 import InternVideo2_5
except ImportError:
    print("failed to load internvideo2_5")
try:
    from video_bench.models.ola_7b import OLA
except ImportError:
    print("failed to load ola")


MULTI_CHOICE_PROMPT = "Answer with the option's letter from the given choices directly."
IV_PROMPT_IMAGE_FIRST = (
    "We provide you with an image placed at the very beginning, followed by a video that has been divided into {frame_num} evenly spaced frames across its {duration} seconds duration. Please answer the question based on the content from both the image and the extracted video frames."
)
IV_PROMPT_VIDEO_FIRST = (
    "We provide you with a video that has been divided into {frame_num} evenly spaced frames across its {duration} seconds duration, followed by an image. Please answer the question based on the content from both the video frames and the image."
)
IV_PROMPT_VIDEO_FIRST_LLaVA_Video = (
    "We provide you with a video followed by an image. Please answer the question based on the content from both the video frames and the image."
)


def format_question(question, options):
    formatted_options = "\n".join([f"{key}. {value}" for key, value in options.items()])
    return f"{question}？\n{formatted_options}"

def convert_to_multiple_choice(data):
    def shuffle_options(correct, distractors):
        options = distractors + [correct]
        random.shuffle(options)
        return options

    def get_option_letter(index):
        return chr(ord('a') + index)

    multiple_choice_questions = []
    for item in data:
        correct_answer = item['answer']
        distractors = [d for d in item.get('distractors', []) if d.strip()]
        all_options = shuffle_options(correct_answer, distractors)
        correct_option_index = all_options.index(correct_answer)
        correct_option_letter = get_option_letter(correct_option_index)
        options = {get_option_letter(i): opt for i, opt in enumerate(all_options)}
        text = format_question(item["question"], options) + "\n" + MULTI_CHOICE_PROMPT
        question_entry = {
            "question": item["question"],
            "data_id": item["data_id"],
            "image_name": item["image_name"],
            "question_type": item["question_type"],
            "granularity": item["granularity"],
            "options": options,
            "text": text,
            "correct_option": correct_option_letter,
            "answer": correct_answer,
        }
        multiple_choice_questions.append(question_entry)
    return multiple_choice_questions

def load_questions_from_jsonl(question_file):
    all_data = {}
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for video_key, questions_list in obj.items():
                formatted_questions = convert_to_multiple_choice(questions_list)
                all_data.setdefault(video_key, []).extend(formatted_questions)
    return all_data

def get_video_duration(video_path):
    command = [
        "ffprobe", 
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        metadata = json.loads(result.stdout)
        return float(metadata['format']['duration'])
    except Exception as e:
        print(f"Failed to get video duration: {e}")
        return None


def check_answer(response, answer):
    all_choices = [chr(i) for i in range(ord('a'), ord('j') + 1)]
    if response is None:
        return False, random.choice(all_choices)
    response = response.strip(" ,.!?;:'").lower()
    response_with_spaces = f" {response} "
    match = re.search(r'<answer>: (\w)', response)
    if match:
        extracted = match.group(1)
        return (extracted == answer), extracted
    candidates = []
    for choice in reversed(all_choices):
        if f"({choice})" in response_with_spaces or f" {choice} " in response_with_spaces or f"{choice}." in response_with_spaces:
            candidates.append(choice)
    pred_choice = candidates[-1] if candidates else random.choice(all_choices)
    return (pred_choice == answer), pred_choice

def load_existing_results(output_file):
    processed_ids = set()
    total_answers = valid_answers = correct_answers = 0
    question_type_stats = defaultdict(lambda: {"total": 0, "valid": 0, "correct": 0, "correct_rate": 0.0})
    question_type_stats["total"] = {"total": 0, "valid": 0, "correct": 0, "correct_rate": 0.0}
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line.strip())
                for video_key, results in result.items():
                    for item in results:
                        processed_ids.add((video_key, item["data_id"]))
                        if item.get("is_true"):
                            correct_answers += 1
                            question_type_stats[item["question_type"]]["correct"] += 1
                            question_type_stats["total"]["correct"] += 1
                        if item.get("model_response") != "null":
                            valid_answers += 1
                            question_type_stats[item["question_type"]]["valid"] += 1
                            question_type_stats["total"]["valid"] += 1
                        total_answers += 1
                        question_type_stats[item["question_type"]]["total"] += 1
                        question_type_stats["total"]["total"] += 1
    for qtype, stats in question_type_stats.items():
        if stats["total"] > 0:
            stats["correct_rate"] = stats["correct"] / stats["total"]
    return processed_ids, total_answers, valid_answers, correct_answers, question_type_stats

def parse_target_pixels(value):
    if value:
        if isinstance(value, str) and ' * ' in value:
            parts = value.split(' * ')
            return int(parts[0]) * int(parts[1])
        else:
            return int(value)
    return None

def build_prompt_text(base_text, duration, nframes, image_pos, model_name):
    if image_pos == "before":
        prompt = IV_PROMPT_IMAGE_FIRST.format(duration=duration, frame_num=int(nframes))
    else:
        prompt = IV_PROMPT_VIDEO_FIRST.format(duration=duration, frame_num=int(nframes))
    if model_name == "llava_video":
        prompt = IV_PROMPT_VIDEO_FIRST_LLaVA_Video
    return prompt + base_text

def load_model_instance(model_name, model_path, max_num):
    from video_bench.registry import get_model
    ModelClass = get_model(model_name)
    kwargs = {"model_path": model_path}
    if max_num is not None:
        kwargs["max_num"] = max_num
    model = ModelClass(**kwargs)
    return model, model_name

def process_question(video_key, question, model, params, prompt_builder, video_dir, image_dir, nframes, all_options):
    data_id = question["data_id"]
    question_text = question["question"]
    choices = question["options"]
    base_text = question["text"]
    correct_option = question["correct_option"]
    question_type = question["question_type"]
    granularity = question["granularity"]

    video_path = os.path.join(video_dir, f"{video_key}.mp4")
    image_path = os.path.join(image_dir, question["image_name"])
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    duration = get_video_duration(video_path)
    prompt_text = prompt_builder(base_text, duration, nframes)
    output = None
    try:
        if params["has_image"]:
            if params["target_resolution"]:
                min_pixels_val = params["min_pixels"] if params["min_pixels"] is not None else 224*224
                max_pixels_val = params["max_pixels"] if params["max_pixels"] is not None else 4096*4096
                output = model.generate_until2(
                    video_path, image_path, prompt_text,
                    (params["target_resolution"][0], params["target_resolution"][1]),
                    params["keep_aspect_ratio"],
                    min_pixels_val, max_pixels_val
                )
            else:
                if params["image_pos"] == "before":
                    output = model.generate_until3(video_path, image_path, prompt_text)
                else:
                    output = model.generate_until1(video_path, image_path, prompt_text)
        else:
            if nframes:
                output = model.generate_video_only(video_path, prompt_text, nframes)
            else:
                output = model.generate_video_only_res(video_path, prompt_text, params["target_resolution"])
    except Exception as e:
        print(f"Error generating output for video {video_key}, data ID {data_id}: {e}")
    is_correct, model_answer = check_answer(output, correct_option)
    print(f"Video: {video_key}, Data ID: {data_id}")
    print(f"Question: {question_text}")
    for label, choice in choices.items():
        print(f"  {label}. {choice}")
    print(f"Model Response: {output}")
    print(f"Correct Answer: {correct_option}. {choices.get(correct_option, '')}")
    print("-" * 50)
    print(is_correct)

    return {
        "data_id": data_id,
        "image_name": question["image_name"],
        "question": question_text,
        "question_type": question_type,
        "granularity": granularity,
        "choices": choices,
        "model_answer": model_answer,
        "correct_option": f"{correct_option}. {choices.get(correct_option, '')}",
        "is_true": is_correct,
        "model_response": output
    }

def main():
    parser = argparse.ArgumentParser(description="Process video questions with a vision-language model using command-line parameters.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    parser.add_argument("--question_file", type=str, required=True, help="Path to the question JSONL file")
    parser.add_argument("--has_image", type=str, default="true", help="Whether to use images (true/false)")
    parser.add_argument("--nframes", type=int, required=True, help="Number of frames to extract from video")
    parser.add_argument("--max_num", type=int, default=None, help="Max num parameter for the model")
    parser.add_argument("--min_pixels", type=str, default=None, help="Minimum pixels in the format 'width * height'")
    parser.add_argument("--max_pixels", type=str, default=None, help="Maximum pixels in the format 'width * height'")
    parser.add_argument("--target_resolution", type=lambda v: ast.literal_eval(v), required=True, help="Target resolution as a tuple, e.g. \"(224, 224)\"")
    parser.add_argument("--keep_aspect_ratio", type=str, default="true", help="Whether to keep aspect ratio (true/false)")
    parser.add_argument("--image_pos", type=str, default="after", help="Position of the image relative to the video ('before' or 'after')")
    args = parser.parse_args()

    # 将部分参数转换为对应类型
    has_image = args.has_image.lower() in ["true", "1", "yes"]
    keep_aspect_ratio = args.keep_aspect_ratio.lower() in ["true", "1", "yes"]

    min_pixels = parse_target_pixels(args.min_pixels)
    max_pixels = parse_target_pixels(args.max_pixels)
    target_resolution = args.target_resolution 

    params = {
        "has_image": has_image,
        "target_resolution": target_resolution,
        "keep_aspect_ratio": keep_aspect_ratio,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "image_pos": args.image_pos,
    }

    print("Loading questions from:", args.question_file)
    question_data = load_questions_from_jsonl(args.question_file)

    processed_ids, total_answers, valid_answers, correct_answers, question_type_stats = load_existing_results(args.output_file)
    print("Processed IDs:", processed_ids)

    model, model_name = load_model_instance(args.model_name, args.model_path, args.max_num)
    model.set_frame_num(args.nframes)
    print(f"Using {args.nframes} frames")
    print(f"Output file: {args.output_file}")

    def prompt_builder(base_text, duration, nframes):
        return build_prompt_text(base_text, duration, nframes, args.image_pos, model_name)

    all_options = set(list("abcdefghij"))
    for video_key, questions in tqdm(question_data.items(), total=len(question_data), desc="Processing Videos"):
        video_path = os.path.join(args.video_dir, f"{video_key}.mp4")
        if not os.path.exists(video_path):
            print(f"Video file does not exist: {video_path}")
            continue

        results = {video_key: []}
        for q in questions:
            if (video_key, q["data_id"]) in processed_ids:
                print(f"Skipping processed video/data_id combination: {video_key}, {q['data_id']}")
                continue
            result = process_question(video_key, q, model, params, prompt_builder, args.video_dir, args.image_dir, args.nframes, all_options)
            if result:
                results[video_key].append(result)
                total_answers += 1
                q_type = q["question_type"]
                question_type_stats["total"]["total"] += 1
                question_type_stats[q_type]["total"] += 1
                if result["model_response"] and result["model_response"] != "null":
                    valid_answers += 1
                    question_type_stats["total"]["valid"] += 1
                    question_type_stats[q_type]["valid"] += 1
                if result["is_true"]:
                    correct_answers += 1
                    question_type_stats["total"]["correct"] += 1
                    question_type_stats[q_type]["correct"] += 1
        if results.get(video_key):
            with open(args.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(results, ensure_ascii=False) + "\n")

    for q_type, stats in question_type_stats.items():
        valid = stats["valid"]
        stats["correct_rate"] = stats["correct"] / valid if valid > 0 else 0.0

    with open(args.output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(question_type_stats, ensure_ascii=False) + "\n")

    print("Statistics:")
    print(f"Total answers: {total_answers}")
    print(f"Valid answers: {valid_answers}")
    print(f"Correct answers: {correct_answers}")
    print("\nBy question type:")
    for q_type, stats in question_type_stats.items():
        print(f"Question type: {q_type}")
        print(f"  Total: {stats['total']}")
        print(f"  Valid: {stats['valid']}")
        print(f"  Correct: {stats['correct']}")
        print(f"  Correct rate: {stats['correct_rate']:.2%}")
        print("-" * 50)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
