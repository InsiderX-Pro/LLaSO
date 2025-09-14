import json
import time
def read_json(file_path):
    """读取 JSON 文件并返回数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(file_path, data):
    """将数据写入到 JSON 文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ 文件已保存: {file_path}")

def extract_labels(data):
    """
    从测试集数据中提取所有标签的集合
    返回一个包含所有标签的集合
    """
    all_labels = set()
    for entry in data:
        true_label = entry['conversations'][1]['value'].replace('.','').lower()
        all_labels.add(true_label)
    return all_labels

def compute_accuracy(data, all_labels):
    """计算模型的准确率：只当模型输出包含当前样本标签而没有其他标签时才算正确"""
    correct = 0
    total = 0
    
    for entry in data:
        true_label = entry['conversations'][1]['value'].replace('.','').lower()
        predicted_label = entry['answer'].replace('.','').lower()
        
        # 判断预测是否和真实标签完全一致
        if true_label == predicted_label:
            correct += 1
        else:
            # 提取模型的输出中是否包含除了当前样本的标签之外的其他标签
            other_labels = all_labels - {true_label}  # 所有标签中去掉当前样本的标签
            
            # 判断模型输出中是否只包含当前标签而没有其他标签
            if true_label in predicted_label and not any(other_label in predicted_label for other_label in other_labels):
                correct += 1
                
        total += 1
        
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

if __name__ == "__main__":
    # 输入和输出文件路径
    test_files = [
        "/code/syr/LLaSO/test/Synthetic_Audio_Classification_test.json",
       "/code/syr/LLaSO/test/velocity_classification_test.json"
    ]
    
    # 遍历文件列表，进行推理和计算准确率
    for test_file in test_files: 
        print(f"正在处理文件: {test_file}")
        
        # 读取数据
        data = read_json(test_file)

        # 获取所有标签集合
        all_labels = extract_labels(data)

        # 计算准确率
        accuracy = compute_accuracy(data, all_labels)
        print(f"文件 {test_file} 准确率: {accuracy * 100:.2f}%")
        
        # 等待1秒
        time.sleep(1)
    
    print("✅ 所有测试完成！")
