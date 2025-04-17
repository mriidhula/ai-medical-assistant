import os
import re
import torch
import platform
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

from langchain.llms import CTransformers
from langchain.chains import create_stuff_documents_chain, create_retrieval_chain
from rag.prompts import prompt
from rag.retriever import get_retriever
from classifier.model import EmotionLandmarkNet
from tts.speak import text_to_speech_with_gtts

# --- Load LLM ---
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type="llama",
    config={"max_new_tokens": 500, "temperature": 0.2}
)

retriever = get_retriever()
qa_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# --- Image classification ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 224 * 224 * 3
model = EmotionLandmarkNet(input_size=input_size, num_classes=2)
model.load_state_dict(torch.load("classifier/wound_classification_model.pth", map_location=device))
model.eval().to(device)

class_names = ['critical', 'non_critical']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
        if confidence.item() < 0.7:
            pred = 1 - pred
        return class_names[pred.item()], confidence.item(), image

def clean_filename(filename):
    return re.sub(r'[^a-zA-Z_]', '', os.path.splitext(os.path.basename(filename))[0])

# --- Main loop ---
if __name__ == "__main__":
    while True:
        print("\nOptions:\n1: Image input\n2: Text input\n3: Both\n4: Exit")
        try:
            choice = int(input("Select (1-4): "))
        except ValueError:
            print("Invalid input.")
            continue

        query = ""
        if choice == 1:
            image_path = input("Enter image path: ")
            if not os.path.exists(image_path):
                print("Image not found.")
                continue
            label, conf, image = predict_image(image_path)
            cleaned_name = clean_filename(image_path)
            query = f"{label}, {cleaned_name}"

            print(f"Predicted: {label} (Conf: {conf:.2f})")
            plt.imshow(image)
            plt.title(f"{label} ({conf:.2f})")
            plt.axis("off")
            plt.show()

        elif choice == 2:
            query = input("Ask a question: ")

        elif choice == 3:
            image_path = input("Enter image path: ")
            if not os.path.exists(image_path):
                print("Image not found.")
                continue
            label, conf, image = predict_image(image_path)
            cleaned_name = clean_filename(image_path)
            plt.imshow(image)
            plt.title(f"{label} ({conf:.2f})")
            plt.axis("off")
            plt.show()
            text = input("Ask a related question: ")
            query = f"{label}, {cleaned_name}, {text}"

        elif choice == 4:
            print("Exiting.")
            break
        else:
            print("Invalid choice.")
            continue

        if query:
            result = qa_chain.invoke({"input": query})
            answer = result["answer"]
            print("\n--- RAG Response ---\n", answer)
            text_to_speech_with_gtts(answer)
