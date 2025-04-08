import os
import json
import requests
import argparse
from tqdm import tqdm
import numpy as np
import time


def load_pope_data(pope_path, pope_type):
    """加载POPE测试数据"""
    pope_file = os.path.join(pope_path, f"coco_pope_{pope_type}.json")

    # 先打印文件的前几行来检查格式
    print(f"Checking format of {pope_file}...")
    try:
        with open(pope_file, "r") as f:
            first_lines = "".join([f.readline() for _ in range(5)])
        print(f"First 5 lines of the file:\n{first_lines}")
    except Exception as e:
        print(f"Error reading file: {e}")

    # 尝试不同的加载方式
    try:
        # 标准JSON解析
        with open(pope_file, "r") as f:
            pope_data = json.load(f)
            return pope_data
    except json.JSONDecodeError as e:
        print(f"Standard JSON parsing failed: {e}")

        # 尝试逐行读取JSON对象
        try:
            with open(pope_file, "r") as f:
                content = f.read()
                # 检查文件是否以JSONL格式存储(每行一个JSON对象)
                if "\n" in content and content.strip().startswith("{"):
                    lines = content.strip().split("\n")
                    pope_data = [json.loads(line) for line in lines if line.strip()]
                    print(f"Loaded {len(pope_data)} items using line-by-line parsing")
                    return pope_data
        except Exception as e2:
            print(f"Line-by-line parsing failed: {e2}")

        # 尝试清理JSON然后解析
        try:
            with open(pope_file, "r") as f:
                content = f.read()
                # 尝试修复一些常见的JSON问题
                content = content.replace("}\n{", "},{")
                content = content.replace("}\r\n{", "},{")

                # 如果不是以[开头，手动添加
                if not content.strip().startswith("["):
                    content = "[" + content

                # 如果不是以]结尾，手动添加
                if not content.strip().endswith("]"):
                    content = content + "]"

                # 尝试解析修复后的内容
                pope_data = json.loads(content)
                print(f"Loaded {len(pope_data)} items after fixing JSON format")
                return pope_data
        except Exception as e3:
            print(f"JSON repair attempt failed: {e3}")

        # 如果所有尝试都失败，打印更多信息并退出
        print(f"All parsing attempts failed for {pope_file}")
        print("Please check the file format manually and ensure it's valid JSON")
        raise ValueError(f"Failed to parse POPE data file: {pope_file}")


def get_image_path(image_filename, coco_path):
    """根据图像文件名获取图像路径"""
    # 文件名已经是完整的格式，直接使用
    image_path = os.path.join(coco_path, "val2014", image_filename)
    return image_path


def query_model(image_path, question, api_url, model_name="qwen2_vl", max_tokens=512):
    """查询Qwen2.5_VL模型"""
    prompt = [
        {"type": "image_url", "image_url": image_path},
        {"type": "text", "text": question},
    ]

    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": False,
        "do_sample": True,
        "repetition_penalty": 1.00,
        "temperature": 0.01,
        "top_p": 0.001,
        "top_k": 1,
        "model": model_name,
    }

    try:
        response = requests.post(api_url, json=data)
        response.raise_for_status()
        return response.json()["text"][0]
    except Exception as e:
        print(f"Error querying model: {e}")
        return None


def parse_yes_no_answer(response):
    """解析模型响应，提取"是"或"否"的答案"""
    # 首先检查完整的"是"或"否"
    response = response.lower()

    # 查找英文回答
    if "yes" in response.lower():
        return "yes"
    elif "no" in response.lower():
        return "no"

    # 查找中文回答
    if "是" in response:
        return "yes"
    elif "否" in response or "不是" in response:
        return "no"

    # 如果没有明确的是/否，则需要进一步分析
    # 这里可以添加更复杂的解析逻辑

    # 默认情况下返回None表示无法确定
    return None


def calculate_metrics(ground_truths, predictions):
    """手动计算评估指标，不依赖sklearn"""
    if not ground_truths or not predictions or len(ground_truths) != len(predictions):
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    total = len(ground_truths)
    correct = sum(1 for gt, pred in zip(ground_truths, predictions) if gt == pred)

    # 计算精确率、召回率和F1分数（针对"yes"作为正类）
    true_positives = sum(
        1
        for gt, pred in zip(ground_truths, predictions)
        if gt == "yes" and pred == "yes"
    )
    false_positives = sum(
        1
        for gt, pred in zip(ground_truths, predictions)
        if gt == "no" and pred == "yes"
    )
    false_negatives = sum(
        1
        for gt, pred in zip(ground_truths, predictions)
        if gt == "yes" and pred == "no"
    )

    accuracy = correct / total if total > 0 else 0

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def evaluate_pope_batch(
    pope_data, coco_path, api_url, output_file, batch_size=50, resume_from=0
):
    """评估POPE测试 - 批量处理版本"""
    # 检查是否有已存在的结果文件用于恢复
    existing_results = []
    existing_predictions = []
    existing_ground_truths = []

    if os.path.exists(output_file) and resume_from > 0:
        try:
            with open(output_file, "r") as f:
                existing_data = json.load(f)
                existing_results = existing_data.get("results", [])

                # 提取已有的预测和真值
                for item in existing_results:
                    existing_predictions.append(item["prediction"])
                    existing_ground_truths.append(item["ground_truth"])

                print(f"Resuming from {len(existing_results)} existing results")
        except Exception as e:
            print(f"Failed to load existing results: {e}")
            existing_results = []
            existing_predictions = []
            existing_ground_truths = []

    # 从指定位置继续处理剩余数据
    remaining_data = pope_data[resume_from:]
    results = existing_results
    predictions = existing_predictions
    ground_truths = existing_ground_truths

    # 分批处理
    total_batches = (len(remaining_data) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(remaining_data))
        batch = remaining_data[batch_start:batch_end]

        print(
            f"Processing batch {batch_idx+1}/{total_batches} (items {resume_from+batch_start+1}-{resume_from+batch_end} of {len(pope_data)})"
        )

        for item in tqdm(batch, desc=f"Batch {batch_idx+1}/{total_batches}"):
            image_filename = item["image"]
            image_path = get_image_path(image_filename, coco_path)
            question = item["text"]
            # 直接使用label字段，不需要转换
            ground_truth = item["label"]

            # 确保图像存在
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} does not exist. Skipping.")
                continue

            # 查询模型
            response = query_model(image_path, question, api_url)
            if response is None:
                print(
                    f"Warning: Failed to get response for image {image_id}. Skipping."
                )
                continue

            # 解析响应
            prediction = parse_yes_no_answer(response)
            if prediction is None:
                print(
                    f"Warning: Could not parse yes/no answer from response for image {image_id}. Skipping."
                )
                continue

            # 记录结果，使用image字段替代image_id
            results.append(
                {
                    "image": image_filename,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "response": response,
                }
            )

            predictions.append(prediction)
            ground_truths.append(ground_truth)

        # 每个批次后保存中间结果
        if len(predictions) > 0:
            # 计算当前指标
            metrics = calculate_metrics(ground_truths, predictions)
            metrics.update(
                {"completed_items": len(results), "total_items": len(pope_data)}
            )

            # 保存中间结果
            with open(output_file, "w") as f:
                json.dump({"results": results, "metrics": metrics}, f, indent=2)

            print(f"Intermediate metrics after batch {batch_idx+1}:")
            print(
                f"  Completed: {len(results)}/{len(pope_data)} ({len(results)/len(pope_data)*100:.2f}%)"
            )
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")

    # 计算最终指标
    if len(predictions) > 0:
        final_metrics = calculate_metrics(ground_truths, predictions)
        final_metrics.update(
            {"completed_items": len(results), "total_items": len(pope_data)}
        )

        # 保存最终结果
        with open(output_file, "w") as f:
            json.dump({"results": results, "metrics": final_metrics}, f, indent=2)

        return final_metrics
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5_VL on POPE dataset")
    parser.add_argument(
        "--pope_path",
        type=str,
        default="/root/Qwen2.5-VL-32b/DataAndTest/data/popeDB/data/POPE-main/data/pope_coco/",
        help="Path to POPE dataset",
    )
    parser.add_argument(
        "--coco_path",
        type=str,
        default="/root/Qwen2.5-VL-32b/DataAndTest/data/popeDB/data/POPE-main/data/coco/",
        help="Path to COCO dataset",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://127.0.0.1:1040/generate",
        help="URL of the Qwen2.5_VL API",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/Qwen2.5-VL-32b/DataAndTest/test_result/PopeTestResult/",
        help="Directory to save results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of items to process in each batch",
    )
    parser.add_argument(
        "--resume_from", type=int, default=0, help="Index to resume from (0-based)"
    )
    parser.add_argument(
        "--pope_type",
        type=str,
        default="all",
        choices=["all", "adversarial", "popular", "random"],
        help="POPE type to evaluate",
    )
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 确定要评估的POPE类型
    pope_types = (
        ["adversarial", "popular", "random"]
        if args.pope_type == "all"
        else [args.pope_type]
    )
    all_metrics = {}

    for pope_type in pope_types:
        print(f"Evaluating POPE {pope_type}...")
        try:
            pope_data = load_pope_data(args.pope_path, pope_type)
            output_file = os.path.join(
                args.output_dir, f"pope_{pope_type}_results.json"
            )

            metrics = evaluate_pope_batch(
                pope_data,
                args.coco_path,
                args.api_url,
                output_file,
                batch_size=args.batch_size,
                resume_from=args.resume_from,
            )

            if metrics:
                all_metrics[pope_type] = metrics

                print(f"POPE {pope_type} final metrics:")
                print(
                    f"  Completed: {metrics['completed_items']}/{metrics['total_items']} ({metrics['completed_items']/metrics['total_items']*100:.2f}%)"
                )
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1 Score: {metrics['f1']:.4f}")
        except Exception as e:
            print(f"Error evaluating POPE {pope_type}: {e}")

    # 保存总体指标
    if all_metrics:
        with open(os.path.join(args.output_dir, "pope_all_metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)

        # 打印总体指标
        print("\nOverall POPE metrics:")
        for pope_type, metrics in all_metrics.items():
            print(f"{pope_type.capitalize()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
