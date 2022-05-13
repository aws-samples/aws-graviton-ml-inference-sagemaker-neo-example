import dlr
import requests
import io
import numpy as np
from PIL import Image

# Model downloaded locally on the container during docker build
labels_filename = '/home/model/labels.txt'
model_dirname = '/home/model'


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)


def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = [label.rstrip() for label in f]
        return labels


def download_file(img_url):
    response = requests.get(img_url)
    img_raw = response.content
    img = Image.open(io.BytesIO(img_raw))

    return img


def preprocess(img):
    img = img.resize((256, 256), Image.Resampling.BILINEAR)
    img = img.crop((16, 16, 240, 240))
    img = img.convert('RGB')
    img = np.array(img).astype(np.float32)

    mean = np.array([0.485, 0.456, 0.406])
    stddev = np.array([0.229, 0.224, 0.225])
    img = ((img / 255.0) - mean) / stddev

    img = np.expand_dims(img, 0)
    img = np.transpose(img, [0, 3, 1, 2]).astype(np.float32)

    return img


def postprocess(scores):
    scores = np.squeeze(scores)
    return softmax(scores)


def top_class(N, probs, labels):
    results = np.argsort(probs)[::-1]

    classes = []
    for i in range(N):
        item = results[i]
        classes.append({
            "class": labels[item],
            "prob": float(probs[item])
        })

    return classes


# Lambda function handler expects an image (URL) that will be used for ML inference
def handler(event, context):
    image = download_file(event["image_url"])
    input_data = preprocess(image)

    model = dlr.DLRModel(model_dirname, 'cpu', 0)

    scores = model.run({"data": input_data})
    probs = postprocess(scores)

    labels = load_labels(labels_filename)
    response = top_class(5, probs, labels)

    return response
