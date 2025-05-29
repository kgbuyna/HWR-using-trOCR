import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    TrainingArguments,
    Trainer
)
from PIL import Image, ImageDraw
import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from datetime import datetime
import sqlite3
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HandwritingDatabase:
    """SQLite database for storing handwriting samples and corrections"""
    
    def __init__(self, db_path: str = "handwriting_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS handwriting_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                ground_truth TEXT NOT NULL,
                predicted_text TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_corrected BOOLEAN DEFAULT FALSE,
                bbox TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_sample(self, image_path: str, ground_truth: str, 
                  predicted_text: str = None, confidence: float = None,
                  bbox: List[int] = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        bbox_str = json.dumps(bbox) if bbox else None
        
        cursor.execute("""
            INSERT INTO handwriting_samples 
            (image_path, ground_truth, predicted_text, confidence, bbox)
            VALUES (?, ?, ?, ?, ?)
        """, (image_path, ground_truth, predicted_text, confidence, bbox_str))
        
        conn.commit()
        conn.close()
        return cursor.lastrowid
    
    def get_training_data(self, limit: int = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT image_path, ground_truth FROM handwriting_samples"
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def update_prediction(self, sample_id: int, predicted_text: str, confidence: float):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE handwriting_samples 
            SET predicted_text = ?, confidence = ?, is_corrected = TRUE
            WHERE id = ?
        """, (predicted_text, confidence, sample_id))
        
        conn.commit()
        conn.close()

class LineDetector:
    """YOLO-based line detection for handwriting"""
    
    def __init__(self):
        cache_dir = os.path.expanduser("./.cache/")
        model_dir = "models--Riksarkivet--yolov9-lines-within-regions-1"
        
        # Find the actual model file in snapshots
        snapshots_dir = os.path.join(cache_dir, model_dir, "snapshots")
        snapshot_hash = os.listdir(snapshots_dir)[0]  # Gets the first snapshot
        model_path = os.path.join(snapshots_dir, snapshot_hash, "model.pt")
        
        try:
            self.model = YOLO(model_path)
        except:
            logger.warning(f"Could not load models--Riksarkivet--yolov9-lines-within-regions-1, using YOLOv8n as fallback")
            self.model = YOLO("yolov8n.pt")
    
    def detect_lines(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect text lines in the image
        Returns list of dictionaries with bbox coordinates and confidence
        """
        results = self.model(image, conf=confidence_threshold)
        
        lines = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    lines.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence)
                    })
        
        # Sort lines by vertical position (top to bottom)
        lines.sort(key=lambda x: x['bbox'][1])
        return lines

class HandwritingDataset(Dataset):
    """Custom dataset for handwriting recognition"""
    
    def __init__(self, image_paths: List[str], texts: List[str], processor):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor

    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and process image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Process image and text
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Tokenize text
        labels = self.processor.tokenizer(
            self.texts[idx],
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        return {
            'pixel_values': pixel_values,
            'labels': labels
        }

class HandwritingOCR:
    """Main OCR class that combines line detection and text recognition"""
    
    def __init__(self, model_name: str = "microsoft/trocr-small-handwritten"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize TrOCR
        self.processor = TrOCRProcessor.from_pretrained(model_name, use_fast=False)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Initialize line detector
        self.line_detector = LineDetector()
        
        # Initialize database
        self.db = HandwritingDatabase()
        
        # Create directories
        os.makedirs("training_data", exist_ok=True)
        os.makedirs("models", exist_ok=True)

    
    def extract_line_images(self, image: np.ndarray, lines: List[Dict]) -> List[Image.Image]:
        """Extract individual line images from detected bounding boxes"""
        line_images = []
        
        for line in lines:
            x1, y1, x2, y2 = line['bbox']
            
            # Add some padding
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            # Extract line image
            line_img = image[y1:y2, x1:x2]
            
            # Convert to PIL Image
            line_pil = Image.fromarray(line_img).convert('RGB')
            line_images.append(line_pil)
        
        return line_images
    
    def recognize_text(self, image: Image.Image) -> Tuple[str, float]:
        """Recognize text from a single line image"""
        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode text
        generated_text = self.processor.batch_decode(
            generated_ids.sequences, 
            skip_special_tokens=True
        )[0]
        
        # Calculate confidence (simplified)
        confidence = torch.softmax(generated_ids.scores[0], dim=-1).max().item()
        
        return generated_text, confidence
    
    def process_image(self, image_path: str) -> Dict:
        """Process a full handwriting image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect lines
        lines = self.line_detector.detect_lines(image)
        print(lines, "lines detected")
        if not lines:
            logger.warning("No lines detected, processing entire image")
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
            text, confidence = self.recognize_text(pil_image)
            return {
                'full_text': text,
                'lines': [{'text': text, 'confidence': confidence, 'bbox': None}]
            }
        
        # Extract line images
        line_images = self.extract_line_images(image, lines)
        
        # Recognize text for each line
        results = []
        full_text = []
        
        for i, (line_img, line_info) in enumerate(zip(line_images, lines)):
            text, confidence = self.recognize_text(line_img)
            
            results.append({
                'text': text,
                'confidence': confidence,
                'bbox': line_info['bbox']
            })
            
            full_text.append(text)
        
        return {
            'full_text': ' '.join(full_text),
            'lines': results
        }
    
    def add_training_sample(self, image_path: str, correct_text: str, 
                          bbox: List[int] = None):
        """Add a corrected sample to the training database"""
        sample_id = self.db.add_sample(image_path, correct_text, bbox=bbox)
        logger.info(f"Added training sample {sample_id}: {correct_text}")
        return sample_id
    
    def fine_tune(self, num_epochs: int = 3, batch_size: int = 2, 
                  learning_rate: float = 5e-5):
        """Fine-tune the model with collected samples"""
        # Get training data
        training_data = self.db.get_training_data()
        
        if len(training_data) < 5:
            logger.warning("Not enough training samples. Need at least 5 samples.")
            return
        
        # Prepare dataset
        image_paths, texts = zip(*training_data)
        dataset = HandwritingDataset(list(image_paths), list(texts), self.processor)
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./models/trocr-finetuned",
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=50,
            evaluation_strategy="no",
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,  # Important for MPS
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.processor.tokenizer,
        )
        
        # Fine-tune
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model()
        self.processor.save_pretrained("./models/trocr-finetuned")
        
        logger.info("Fine-tuning completed!")
    
    def interactive_correction(self, image_path: str):
        """Interactive correction workflow"""
        print(f"\nProcessing: {image_path}")
        
        # Process image
        results = self.process_image(image_path)
        
        print(f"Detected text: {results['full_text']}")
        print(f"Confidence: {results['lines'][0]['confidence']:.3f}")
        
        # Ask for correction
        correct_text = input("Enter correct text (or press Enter if correct): ").strip()
        
        if correct_text:
            # Add to training data
            self.add_training_sample(image_path, correct_text)
            print("Added to training data!")
        else:
            print("No correction needed.")
        
        return correct_text if correct_text else results['full_text']

def main():
    """Example usage"""
    # Initialize OCR system
    ocr = HandwritingOCR()
    
    # Example: Process a single image
    # result = ocr.process_image("sample_handwriting.jpg")
    # print(f"Recognized text: {result['full_text']}")
    
    # Example: Interactive correction
    # ocr.interactive_correction("sample_handwriting.jpg")
    
    # Example: Fine-tune after collecting samples
    # ocr.fine_tune(num_epochs=3)
    
    print("OCR system initialized. Ready for use!")

if __name__ == "__main__":
    main()