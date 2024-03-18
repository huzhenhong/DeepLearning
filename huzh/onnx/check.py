from pprint import pprint
import onnxruntime
import numpy as np
import cv2 as cv
from PIL import Image

# onnx_path = "/Users/huzh/Downloads/yolov8s.onnx"
onnx_path = "/Users/huzh/Downloads/person_car_v8s.onnx"

provider = "CPUExecutionProvider"
onnx_session = onnxruntime.InferenceSession(onnx_path, providers=[provider])

print("----------------- 输入部分 -----------------")
input_tensors = onnx_session.get_inputs()  # 该 API 会返回列表
for input_tensor in input_tensors:  # 因为可能有多个输入，所以为列表
    input_info = {
        "name": input_tensor.name,
        "type": input_tensor.type,
        "shape": input_tensor.shape,
    }
    pprint(input_info)

print("----------------- 输出部分 -----------------")
output_tensors = onnx_session.get_outputs()  # 该 API 会返回列表
for output_tensor in output_tensors:  # 因为可能有多个输出，所以为列表
    output_info = {
        "name": output_tensor.name,
        "type": output_tensor.type,
        "shape": output_tensor.shape,
    }
    pprint(output_info)


# 推理的图片路径
image = Image.open(
    '/Users/huzh/Documents/algorithm/物品搬移/test_img/物品搬移/1701163301253.jpg'
).convert('RGB')

# import onnxruntime

# ort_session = onnxruntime.InferenceSession("super_resolution.onnx")


# 将张量转化为ndarray格式
def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy()
        if tensor.requires_grad
        else tensor.cpu().numpy()
    )


image = np.array(image)
image = cv.resize(image, (640, 640))
# .transpose(2, 0, 1)
print(image.shape)
# 构建输入的字典和计算输出结果

import torchvision.transforms as transforms

to_tensor = transforms.ToTensor()
img_y = to_tensor(image)
img_y.unsqueeze_(0)
print("img_y: ", img_y.shape)
# 构建输入的字典并将value转换位array格式
ort_inputs = {onnx_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = onnx_session.run(None, ort_inputs)

print("onnx weights", ort_outs)
print("out shape: ", len(ort_outs))
# print("onnx prediction", outs.argmax(axis=1)[0])
