import torch
from torchvision import transforms
from PIL import Image
import cv2
from OpenScreen.settings import load_settings
from OpenScreen.simpleUnet import SimpleUnet


class GenerateBackgroundReplacement():
    def __init__(self):
        self.settings = load_settings()
        self.frame = None
        self.mask = None
        self.running = False

        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.4117, 0.5926, 0.3815], [0.3299, 0.3250, 0.3212])
        ])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleUnet()
        state_dictionary = torch.load(self.settings["general"]["model"], map_location=self.device)
        self.model.load_state_dict(state_dictionary)
        self.model.to(self.device)
        self.model.eval()  # Set the model to inference mode

    def set_frame(self, frame):
        self.frame = frame
        self.process()

    def get_mask(self):
        return self.mask

    def process(self):
        if self.frame is not None:
            frame = cv2.resize(self.frame, (0, 0), fx=1, fy=1)
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = self.image_transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                binary_mask = self.model(image_tensor) > 0.5

            binary_mask_np = binary_mask.squeeze().cpu().numpy().astype('uint8')
            binary_mask_resized = cv2.resize(binary_mask_np, (self.frame.shape[1], self.frame.shape[0]))

            self.mask = binary_mask_resized
