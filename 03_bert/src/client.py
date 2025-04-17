import requests
import time


# 定义请求url和传入的data
url = "http://127.0.0.1:5000/"
data = {"uid": "2046", "text": "量子计算技术突破，全球科技界迎来新一轮变革"}

start_time = time.time()
# 向服务发送post请求
res = requests.post(url, data=data)
cost_time = time.time() - start_time

# 打印返回结果
print("文本类别: ", res.text)
print("耗时: ", cost_time * 1000, "ms")