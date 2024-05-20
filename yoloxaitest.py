from PIL import Image
from ultralytics import YOLO
import torchvision
from easy_explain import YOLOv8LRP

image = Image.open(r'testenv\131659.jpg')

desired_size = (640, 640)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(desired_size),
    torchvision.transforms.ToTensor(),
])
image = transform(image)

model = YOLO('models/best232.pt')

lrp = YOLOv8LRP(model, power=1, eps=1, device='cpu')
explanation_lrp = lrp.explain(image, cls='0', contrastive=False).cpu()

lrp.plot_explanation(frame=image,
                    explanation = explanation_lrp,
                    contrastive=False,
                    cmap='Reds',
                    title='Explanation for Class "test"'
                    )
