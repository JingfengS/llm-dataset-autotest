#!/bin/bash

# 定义变量
POPE_PATH="/root/Qwen2.5-VL-32b/DataAndTest/data/popeDB/data/POPE-main/data/pope_coco/"
COCO_PATH="/root/Qwen2.5-VL-32b/DataAndTest/data/popeDB/data/POPE-main/data/coco/"
API_URL="http://127.0.0.1:1040/generate"
OUTPUT_DIR="/root/Qwen2.5-VL-32b/DataAndTest/test_result/PopeTestResult/"
BATCH_SIZE=50  # 可以根据需要调整批处理大小
LOG_DIR="${OUTPUT_DIR}/logs"  # 日志目录

# 创建日志目录
mkdir -p ${LOG_DIR}

# 获取当前时间作为日志文件名的一部分
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 定义POPE测试类型
POPE_TYPES=("adversarial" "popular" "random")

# 创建一个函数来检查之前的结果并确定从哪里继续
get_resume_index() {
    local pope_type=$1
    local result_file="${OUTPUT_DIR}/pope_${pope_type}_results.json"
    
    if [ -f "$result_file" ]; then
        # 使用grep和awk提取已完成的项目数
        local completed=$(grep -o '"completed_items": [0-9]*' "$result_file" | awk '{print $2}')
        
        # 如果没有找到，设置为0
        if [ -z "$completed" ]; then
            echo 0
        else
            echo $completed
        fi
    else
        echo 0
    fi
}

# 为每种类型运行测试
for pope_type in "${POPE_TYPES[@]}"; do
    echo "===========================================" | tee -a "${LOG_DIR}/pope_${pope_type}_${TIMESTAMP}.log"
    echo "开始测试 POPE ${pope_type} 数据集..." | tee -a "${LOG_DIR}/pope_${pope_type}_${TIMESTAMP}.log"
    
    # 检查是否有之前的结果，确定从哪里继续
    resume_from=$(get_resume_index "$pope_type")
    
    echo "从索引 ${resume_from} 继续测试..." | tee -a "${LOG_DIR}/pope_${pope_type}_${TIMESTAMP}.log"
    
    # 运行Python脚本
    python pope_evaluate.py \
        --pope_path ${POPE_PATH} \
        --coco_path ${COCO_PATH} \
        --api_url ${API_URL} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size ${BATCH_SIZE} \
        --resume_from ${resume_from} \
        --pope_type ${pope_type} \
        2>&1 | tee -a "${LOG_DIR}/pope_${pope_type}_${TIMESTAMP}.log"
    
    echo "POPE ${pope_type} 测试完成！" | tee -a "${LOG_DIR}/pope_${pope_type}_${TIMESTAMP}.log"
    echo "===========================================" | tee -a "${LOG_DIR}/pope_${pope_type}_${TIMESTAMP}.log"
    echo "" | tee -a "${LOG_DIR}/pope_${pope_type}_${TIMESTAMP}.log"
done

# 生成简单的总结报告
echo "所有POPE测试已完成！结果保存在 ${OUTPUT_DIR} 目录下" | tee -a "${LOG_DIR}/pope_all_${TIMESTAMP}.log"

# 提取每个测试类型的结果并简单汇总
echo "生成总结报告..." | tee -a "${LOG_DIR}/pope_all_${TIMESTAMP}.log"
echo "POPE测试总结 (${TIMESTAMP})" > "${OUTPUT_DIR}/pope_summary_${TIMESTAMP}.txt"
echo "=======================================" >> "${OUTPUT_DIR}/pope_summary_${TIMESTAMP}.txt"

for pope_type in "${POPE_TYPES[@]}"; do
    result_file="${OUTPUT_DIR}/pope_${pope_type}_results.json"
    if [ -f "$result_file" ]; then
        echo "" >> "${OUTPUT_DIR}/pope_summary_${TIMESTAMP}.txt"
        echo "POPE ${pope_type} 结果:" >> "${OUTPUT_DIR}/pope_summary_${TIMESTAMP}.txt"
        echo "--------------------------------------" >> "${OUTPUT_DIR}/pope_summary_${TIMESTAMP}.txt"
        
        # 使用grep提取指标
        accuracy=$(grep -o '"accuracy": [0-9.]*' "$result_file" | awk '{print $2}')
        precision=$(grep -o '"precision": [0-9.]*' "$result_file" | awk '{print $2}')
        recall=$(grep -o '"recall": [0-9.]*' "$result_file" | awk '{print $2}')
        f1=$(grep -o '"f1": [0-9.]*' "$result_file" | awk '{print $2}')
        completed=$(grep -o '"completed_items": [0-9]*' "$result_file" | awk '{print $2}')
        total=$(grep -o '"total_items": [0-9]*' "$result_file" | awk '{print $2}')
        
        echo "完成度: ${completed}/${total}" >> "${OUTPUT_DIR}/pope_summary_${TIMESTAMP}.txt"
        echo "准确率: ${accuracy}" >> "${OUTPUT_DIR}/pope_summary_${TIMESTAMP}.txt"
        echo "精确率: ${precision}" >> "${OUTPUT_DIR}/pope_summary_${TIMESTAMP}.txt"
        echo "召回率: ${recall}" >> "${OUTPUT_DIR}/pope_summary_${TIMESTAMP}.txt"
        echo "F1分数: ${f1}" >> "${OUTPUT_DIR}/pope_summary_${TIMESTAMP}.txt"
    else
        echo "" >> "${OUTPUT_DIR}/pope_summary_${TIMESTAMP}.txt"
        echo "POPE ${pope_type}: 未找到结果文件" >> "${OUTPUT_DIR}/pope_summary_${TIMESTAMP}.txt"
    fi
done

echo "测试总结已保存到 ${OUTPUT_DIR}/pope_summary_${TIMESTAMP}.txt" | tee -a "${LOG_DIR}/pope_all_${TIMESTAMP}.log"
