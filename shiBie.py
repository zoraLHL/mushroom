from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import onnxruntime as rt
import torchvision.transforms as transforms

app = Flask(__name__)
session = rt.InferenceSession("./mushroom_test_resnet50.onnx")  # 加载ONNX模型
idx_to_labels = np.load('./model/idx_to_labels.npy', allow_pickle=True).item()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST" ,"GET"])
def predict():
    try:
        # img = request.files["image"]
        img = "./image.jpg"
        image = Image.open(img).convert("RGB")
        image = image.resize((256, 256))

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])

            input_img = test_transform(image)
            input_tensor = input_img.unsqueeze(0).numpy()

            input_name = session.get_inputs()[0].name  # 获取输入名称
            output_name = session.get_outputs()[0].name  # 获取输出名称

            # 获取预测结果
            result = session.run([output_name], {input_name: input_tensor})
            prediction = result[0]

            predicted_class_id = np.argmax(prediction)  # 获取预测类别ID
            predicted_class = idx_to_labels[predicted_class_id]  # 获取预测类别名称

            return jsonify(predicted_class)
        except Exception as e:
            print(e)
            return jsonify(msg="error！")

    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=5000)
