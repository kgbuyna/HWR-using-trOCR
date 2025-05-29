"""
Command Line Interactive Demo for Handwriting OCR System
No GUI dependencies - works entirely in terminal
"""

import os
import sys
from pathlib import Path
import argparse
import glob
from PIL import Image
import cv2
import numpy as np

# Import our OCR system
from handwriting_ocr_pipeline import HandwritingOCR

class CLIHandwritingOCR:
    def __init__(self):
        self.ocr = HandwritingOCR()
        print("🔥 Handwriting OCR System Initialized")
        print("📱 Ready for mobile app integration pipeline")
    
    def process_single_image(self, image_path: str, interactive: bool = True):
        """Process a single image with optional interactive correction"""
        if not os.path.exists(image_path):
            print(f"❌ Error: Image not found: {image_path}")
            return None
        
        print(f"\n📷 Processing: {os.path.basename(image_path)}")
        print("⏳ Running OCR...")
        
        try:
            # Process with OCR
            results = self.ocr.process_image(image_path)
            
            # Display results
            print("\n" + "="*60)
            print("📝 RECOGNIZED TEXT:")
            print("="*60)
            print(f"{results['full_text']}")
            print("="*60)
            
            # Show line-by-line results
            if len(results['lines']) > 1:
                print("\n📋 Line by Line Results:")
                for i, line in enumerate(results['lines'], 1):
                    conf_emoji = "🟢" if line['confidence'] > 0.8 else "🟡" if line['confidence'] > 0.5 else "🔴"
                    print(f"  {i}. {line['text']} {conf_emoji} ({line['confidence']:.3f})")
            
            # Interactive correction
            if interactive:
                return self.interactive_correction(image_path, results['full_text'])
            else:
                return results['full_text']
                
        except Exception as e:
            print(f"❌ Processing failed: {str(e)}")
            return None
    
    def interactive_correction(self, image_path: str, recognized_text: str):
        """Interactive correction in command line"""
        print(f"\n🔧 CORRECTION MODE")
        print("="*40)
        print("Options:")
        print("  1. Press ENTER if text is correct")
        print("  2. Type correct text if there are errors")
        print("  3. Type 'skip' to skip this image")
        print("="*40)
        
        user_input = input("Enter correction (or press ENTER if correct): ").strip()
        
        if user_input.lower() == 'skip':
            print("⏭️  Skipped")
            return recognized_text
        elif user_input == '':
            print("✅ Text marked as correct")
            return recognized_text
        else:
            try:
                # Add corrected sample to training data
                sample_id = self.ocr.add_training_sample(image_path, user_input)
                print(f"✅ Correction added to training data (ID: {sample_id})")
                return user_input
            except Exception as e:
                print(f"❌ Error saving correction: {str(e)}")
                return user_input
    
    def batch_process(self, directory: str, pattern: str = "*.jpg"):
        """Process multiple images in a directory"""
        search_pattern = os.path.join(directory, pattern)
        image_files = glob.glob(search_pattern)
        
        # Also search for other common formats
        for ext in ["*.png", "*.jpeg", "*.bmp", "*.tiff"]:
            if ext != pattern:
                image_files.extend(glob.glob(os.path.join(directory, ext)))
        
        if not image_files:
            print(f"❌ No images found in {directory} with pattern {pattern}")
            return
        
        print(f"📁 Found {len(image_files)} images to process")
        
        results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"\n📊 Progress: {i}/{len(image_files)}")
            result = self.process_single_image(image_path, interactive=True)
            if result:
                results.append({
                    'image': os.path.basename(image_path),
                    'text': result
                })
        
        return results
    
    def fine_tune_model(self):
        """Fine-tune the model with collected corrections"""
        # Check how many training samples we have
        training_data = self.ocr.db.get_training_data()
        num_samples = len(training_data)
        
        print(f"\n🎯 FINE-TUNING MODE")
        print("="*40)
        print(f"📊 Training samples available: {num_samples}")
        
        if num_samples < 5:
            print("⚠️  Warning: Less than 5 samples. Recommended minimum is 10-20.")
            print("   You can continue, but results may not be optimal.")
        
        proceed = input("\n🚀 Start fine-tuning? (y/N): ").lower()
        
        if proceed == 'y':
            try:
                print("⏳ Fine-tuning in progress...")
                print("   This may take several minutes...")
                
                # Fine-tune with conservative settings for small datasets
                epochs = 3 if num_samples >= 10 else 2
                batch_size = 1  # Small batch for M1 Mac
                learning_rate = 5e-5 if num_samples >= 10 else 1e-5
                
                self.ocr.fine_tune(
                    num_epochs=epochs, 
                    batch_size=batch_size, 
                    learning_rate=learning_rate
                )
                
                print("✅ Fine-tuning completed!")
                print("📱 Model ready for mobile app integration")
                
            except Exception as e:
                print(f"❌ Fine-tuning failed: {str(e)}")
        else:
            print("❌ Fine-tuning cancelled")
    
    def show_training_stats(self):
        """Show training data statistics"""
        training_data = self.ocr.db.get_training_data()
        
        print(f"\n📊 TRAINING DATA STATISTICS")
        print("="*40)
        print(f"Total samples: {len(training_data)}")
        
        if training_data:
            # Show recent samples
            print(f"\n📝 Recent corrections:")
            for i, (image_path, text) in enumerate(training_data[-5:], 1):
                filename = os.path.basename(image_path)
                preview = text[:50] + "..." if len(text) > 50 else text
                print(f"  {i}. {filename}: {preview}")
    
    def export_training_data(self, output_file: str = "training_export.txt"):
        """Export training data for mobile app"""
        training_data = self.ocr.db.get_training_data()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Handwriting OCR Training Data Export\n")
            f.write(f"# Total samples: {len(training_data)}\n\n")
            
            for image_path, text in training_data:
                f.write(f"Image: {image_path}\n")
                f.write(f"Text: {text}\n")
                f.write("-" * 50 + "\n")
        
        print(f"✅ Training data exported to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Handwriting OCR CLI Demo')
    parser.add_argument('command', choices=['process', 'batch', 'finetune', 'stats', 'export'], 
                       help='Command to execute')
    parser.add_argument('--image', '-i', help='Path to image file')
    parser.add_argument('--directory', '-d', help='Directory containing images')
    parser.add_argument('--pattern', '-p', default='*.jpg', help='File pattern for batch processing')
    parser.add_argument('--output', '-o', help='Output file for export')
    parser.add_argument('--no-interactive', action='store_true', help='Disable interactive correction')
    
    args = parser.parse_args()
    
    # Initialize CLI OCR
    print(args.command.upper(), "COMMAND SELECTED")
    cli_ocr = CLIHandwritingOCR()
    if args.command == 'process':
        if not args.image:
            print("❌ Error: --image required for process command")
            sys.exit(1)
        
        result = cli_ocr.process_single_image(args.image, interactive=not args.no_interactive)
        
    elif args.command == 'batch':
        if not args.directory:
            print("❌ Error: --directory required for batch command")
            sys.exit(1)
        
        results = cli_ocr.batch_process(args.directory, args.pattern)
        
        # Save batch results
        if results:
            output_file = "batch_results.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"Image: {result['image']}\n")
                    f.write(f"Text: {result['text']}\n")
                    f.write("-" * 50 + "\n")
            print(f"📄 Batch results saved to: {output_file}")
    
    elif args.command == 'finetune':
        cli_ocr.fine_tune_model()
    
    elif args.command == 'stats':
        cli_ocr.show_training_stats()
    
    elif args.command == 'export':
        output_file = args.output or "training_export.txt"
        cli_ocr.export_training_data(output_file)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode if no arguments
        print("🚀 HANDWRITING OCR - Interactive Mode")
        print("="*50)
        
        cli_ocr = CLIHandwritingOCR()
        
        while True:
            print(f"\n📋 MAIN MENU")
            print("="*30)
            print("1. Process single image")
            print("2. Batch process directory")
            print("3. Fine-tune model")
            print("4. Show training stats")
            print("5. Export training data")
            print("6. Exit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                image_path = input("Enter image path: ").strip()
                cli_ocr.process_single_image(image_path)
            
            elif choice == '2':
                directory = input("Enter directory path: ").strip()
                pattern = input("File pattern (default: *.jpg): ").strip() or "*.jpg"
                cli_ocr.batch_process(directory, pattern)
            
            elif choice == '3':
                cli_ocr.fine_tune_model()
            
            elif choice == '4':
                cli_ocr.show_training_stats()
            
            elif choice == '5':
                output_file = input("Output file (default: training_export.txt): ").strip() or "training_export.txt"
                cli_ocr.export_training_data(output_file)
            
            elif choice == '6':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice")
    else:
        main()