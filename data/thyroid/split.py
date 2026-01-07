import json

def split_dataset(input_file, train_file, val_file):
    """
    将数据集按照split字段分割成训练集和验证集
    split=3和split=4的数据作为验证集，其余作为训练集
    """
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    train_lines = []
    val_lines = []
    
    for line in lines:
        data = json.loads(line.strip())
        split_value = data.get('split')
        
        # split=3和split=4作为验证集，其余作为训练集
        if split_value in ['3']:
            val_lines.append(line)
        else:
            train_lines.append(line)
    
    # 写入训练集文件
    with open(train_file, 'w', encoding='utf-8') as f_train:
        f_train.writelines(train_lines)
    
    # 写入验证集文件
    with open(val_file, 'w', encoding='utf-8') as f_val:
        f_val.writelines(val_lines)
    
    print(f"数据分割完成:")
    print(f"训练集数量: {len(train_lines)}")
    print(f"验证集数量: {len(val_lines)}")
    print(f"总数据量: {len(lines)}")

# 使用示例
input_file = '/tmp/shared-storage/lishichao/EasyR1/data/thyroid/all.jsonl'
train_file = '/tmp/shared-storage/lishichao/EasyR1/data/thyroid/folds/3/train.jsonl'
val_file = '/tmp/shared-storage/lishichao/EasyR1/data/thyroid/folds/3/val.jsonl'

split_dataset(input_file, train_file, val_file)