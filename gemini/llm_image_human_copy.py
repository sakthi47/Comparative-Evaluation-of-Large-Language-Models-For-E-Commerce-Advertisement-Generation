#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime
import os
from base_generator import GeminiHuggingFaceAdGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate Hugging Face image from human-provided description using Gemini optimization')
    parser.add_argument('--description', '-d', required=True, help='Product description for image generation')
    parser.add_argument('--style', '-s', default='modern minimalist', help='Brand/visual style')
    parser.add_argument('--copy', '-c', default='', help='Ad copy context for image generation')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--model', '-m', default='flux-schnell', help='HuggingFace image model (flux-schnell, stable-diffusion, playground)')
    parser.add_argument('--gemini-model', default='gemini-1.5-flash', help='Gemini model for prompt optimization')
    parser.add_argument('--output', '-o', help='Output JSON filename')
    parser.add_argument('--download', action='store_true', help='Download generated image')
    
    args = parser.parse_args()
    
    try:
        os.makedirs("./output/llm_image_human_copy", exist_ok=True)
        
        generator = GeminiHuggingFaceAdGenerator(model_name=args.gemini_model)
        
        print(f"Generating optimized image prompt for: {args.description}")
        
        # Step 1: Use Gemini to generate optimized image prompt
        optimized_prompt = generator.generate_image_prompt(
            product_description=args.description,
            brand_style=args.style,
            ad_copy=args.copy
        )
        
        print(f"Gemini-optimized prompt: {optimized_prompt}")
        
        # Step 2: Generate image with Hugging Face
        print("Generating image with Hugging Face...")
        image_result = generator.generate_ad_image(
            text_prompt=optimized_prompt,
            image_model=args.model,
            width=args.width,
            height=args.height
        )
        
        if 'error' in image_result:
            print(f"Error: {image_result['error']}")
            sys.exit(1)
        
        results = {
            "condition": "LI (LLM Image / Human Copy)",
            "timestamp": datetime.now().isoformat(),
            "product_description": args.description,
            "brand_style": args.style,
            "ad_copy_context": args.copy,
            "gemini_model": args.gemini_model,
            "gemini_optimized_prompt": optimized_prompt,
            "image_generation_result": image_result,
            "image_settings": {
                "model": args.model,
                "width": args.width,
                "height": args.height
            }
        }
        
        # Always save the image by default
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f"./output/llm_image_human_copy/li_generated_image_{timestamp}.jpg"
        
        try:
            saved_path = generator.download_image(image_result['image_object'], image_filename)
            results["downloaded_image"] = saved_path
            print(f"Image saved to: {saved_path}")
        except Exception as e:
            print(f"Warning: Could not save image - {e}")
            if args.download:
                print("Image generation succeeded but saving failed")
        
        output_file = args.output or f"./output/llm_image_human_copy/li_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generator.save_results(results, output_file)
        
        print(f"Gemini-Optimized Prompt: {optimized_prompt}")
        print(f"Image Model Used: {args.model}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()