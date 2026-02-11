from transformers import pipeline
from PIL import Image

pipe = pipeline(
    "image-to-text",
    model="nlpconnect/vit-gpt2-image-captioning"
)

print(pipe(r"C:\Users\INFOMERICA-1127\Documents\MEDICAL CHATBOT PROJECT\DATA\Brain_Cancer raw MRI data\Brain_Cancer\brain_glioma\brain_glioma_0001.jpg"))
