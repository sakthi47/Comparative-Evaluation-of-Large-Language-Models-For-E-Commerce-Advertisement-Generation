#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime
import os
from base_generator import DeepSeekHuggingFaceAdGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate both DeepSeek V3 ad copy and Hugging Face image')
    parser.add_argument('--description', '-d', required=True, help='Product description')
    parser.add_argument('--brand-style', '-b', default='modern', help='Brand style')
    parser.add_argument('--target-audience', '-t', default='general', help='Target audience')
    parser.add_argument('--deepseek-model', default='DeepSeek-V3-0324', help='DeepSeek model for text generation')
    parser.add_argument('--image-model', default='flux-schnell', help='HuggingFace image model')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--output', '-o', help='Output JSON filename')
    parser.add_argument('--download', action='store_true', help='Download generated image (default: True)')
    
    args = parser.parse_args()
    
    try:
        os.makedirs("./output/llm_both_generated", exist_ok=True)
        
        generator = DeepSeekHuggingFaceAdGenerator(model_name=args.deepseek_model)
        
        print(f"Generating complete ad for: {args.description}")
        
        # Step 1: Generate ad copy with DeepSeek V3
        print("Step 1: Generating ad copy with DeepSeek V3...")
        copy_result = generator.generate_ad_copy_from_text(
            product_description=args.description,
            brand_style=args.brand_style,
            target_audience=args.target_audience
        )
        
        if 'error' in copy_result:
            print(f"Error generating copy: {copy_result['error']}")
            sys.exit(1)
        
        primary_copy = copy_result['copy_data'].get('primary_copy', '')
        
        # Step 2: Generate optimized image prompt with DeepSeek V3
        print("Step 2: Generating optimized image prompt with DeepSeek V3...")
        optimized_prompt = generator.generate_image_prompt(
            product_description=args.description,
            brand_style=args.brand_style,
            ad_copy=primary_copy
        )
        
        # Step 3: Generate image with Hugging Face
        print("Step 3: Generating image with Hugging Face...")
        image_result = generator.generate_ad_image(
            text_prompt=optimized_prompt,
            image_model=args.image_model,
            width=args.width,
            height=args.height
        )
        
        if 'error' in image_result:
            print(f"Error generating image: {image_result['error']}")
            sys.exit(1)
        
        results = {
            "condition": "LCI (LLM Copy & LLM Image)",
            "timestamp": datetime.now().isoformat(),
            "product_description": args.description,
            "brand_style": args.brand_style,
            "target_audience": args.target_audience,
            "deepseek_model": args.deepseek_model,
            "copy_generation_result": copy_result,
            "deepseek_optimized_prompt": optimized_prompt,
            "image_generation_result": image_result,
            "image_settings": {
                "model": args.image_model,
                "width": args.width,
                "height": args.height
            }
        }
        
        # Always save the image by default
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f"./output/llm_both_generated/lci_generated_image_{timestamp}.png"
        
        try:
            saved_path = generator.download_image(image_result['image_object'], image_filename)
            results["downloaded_image"] = saved_path
            print(f"Image saved to: {saved_path}")
        except Exception as e:
            print(f"Warning: Could not save image - {e}")
        
        output_file = args.output or f"./output/llm_both_generated/lci_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generator.save_results(results, output_file)
        
        print("\nGenerated Ad Copy:")
        copy_data = copy_result['copy_data']
        print(f"Primary Copy: {copy_data.get('primary_copy', 'N/A')}")
        print(f"Headline: {copy_data.get('headline', 'N/A')}")
        print(f"Description: {copy_data.get('description', 'N/A')}")
        print(f"CTA: {copy_data.get('cta', 'N/A')}")
        
        print(f"\nDeepSeek V3-Optimized Image Prompt: {optimized_prompt}")
        print(f"Image Model Used: {args.image_model}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()