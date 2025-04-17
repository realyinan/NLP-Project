import fasttext


train_data_path = "./data/train_new.txt"
test_data_path = "./data/test_new.txt"

# 开启模型训练
model = fasttext.train_supervised(input=train_data_path, wordNgrams=2)
print("词的数量: ", len(model.words))
print("标签值: ", model.labels)

# 开启模型测试
result = model.test(test_data_path)
print(result)


# 优化: 自动化参数搜索
train_data_path = "./data/train_new.txt"
test_data_path = "./data/test_new.txt"

model = fasttext.train_supervised(train_data_path, wordNgrams=2, autotuneValidationFile=test_data_path, autotuneDuration=20, verbose=3)
result = model.test(test_data_path)
# print(result)

# 模型保存
model.save_model("./data/fasttest_model.bin")
