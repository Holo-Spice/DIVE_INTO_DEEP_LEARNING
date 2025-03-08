import cv2
import torch
import time
import torchvision.transforms as transforms
from PIL import Image


def load_model():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
        torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
        torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
        torch.nn.Flatten(),
        torch.nn.Linear(128 * 3 * 3, 512), torch.nn.ReLU(),
        torch.nn.Linear(512, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )
    model.load_state_dict(torch.load('mnist_cnn.pth', map_location=torch.device('cuda:0')))
    model.eval()
    return model


def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    return transform(Image.fromarray(image).convert('L')).unsqueeze(0)


def predict(model, image):
    with torch.no_grad():
        return torch.argmax(model(preprocess(image)), dim=1).item()


def main():
    net = load_model()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Camera initialization failed")

    roi_size = 280
    try:
        while True:
            start = time.perf_counter()
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            x = max(0, (w - roi_size) // 2)
            y = max(0, (h - roi_size) // 2)
            adjusted_size = min(roi_size, w - x, h - y)

            roi = gray[y:y + adjusted_size, x:x + adjusted_size]
            cv2.imshow('frame', roi)
            processed = cv2.threshold(cv2.resize(roi, (28, 28)), 100, 255, cv2.THRESH_BINARY_INV)[1]

            try:
                pred = predict(net, processed)
            except Exception as e:
                print(f"Prediction error: {e}")
                pred = -1

            cv2.rectangle(frame, (x, y), (x + adjusted_size, y + adjusted_size), (0, 255, 0), 2)
            cv2.putText(frame, f'Pred: {pred}', (x + 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'Time: {(time.perf_counter() - start) * 1000:.1f}ms', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow('Processed', processed)
            cv2.imshow('Live', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()