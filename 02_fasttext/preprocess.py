# 打开数据集类别文件
id_to_label = {}
idx = 0
with open("./data/class.txt", encoding="utf-8") as f:
    for line in f.readlines():
        line = line.strip()
        id_to_label[idx] = line
        idx += 1
print(id_to_label)


# 标签转化
train_data = []
with open("./data/train.txt", encoding="utf-8") as f:
    for line in f.readlines():
        line = line.strip()
        sentence, label = line.split("\t")
        label_id = int(label)
        label_name = id_to_label[label_id]
        new_label = "__label__" + label_name

        # 文本处理
        sent_char = " ".join(list(sentence))
        new_sentence = new_label + " " + sent_char
        train_data.append(new_sentence)

print(train_data[:3])

# # 将处理后的结果存储在txt文本中
with open("./data/test_new.txt", mode="w", encoding="utf-8") as f:
    for data in train_data:
        f.write(data + "\n")
print("训练数据预处理完毕")