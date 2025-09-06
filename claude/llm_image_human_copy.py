#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime
import os
from base_generator import ClaudeAzureDALLEAdGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate Azure DALL-E image from human-provided description using Claude optimization')
    parser.add_argument('--description', '-d', required=True, help='Product description for image generation')
    parser.add_argument('--style', '-s', default='modern minimalist', help='Brand/visual style')
    parser.add_argument('--copy', '-c', default='', help='Ad copy context for image generation')
    parser.add_argument('--size', default='1024x1024', help='Image size (1024x1024, 1024x1792, 1792x1024)')
    parser.add_argument('--quality', default='standard', help='Image quality (standard, hd)')
    parser.add_argument('--model', '-m', default='dall-e-3', help='DALL-E model deployment name')
    parser.add_argument('--output', '-o', help='Output JSON filename')
    parser.add_argument('--download', action='store_true', help='Download generated image')
    
    args = parser.parse_args()
    
    try:
        os.makedirs("./output/llm_image_human_copy", exist_ok=True)
        
        generator = ClaudeAzureDALLEAdGenerator()
        
        print(f"Generating optimized image prompt for: {args.description}")
        
        # Step 1: Use Claude to generate optimized DALL-E prompt
        dalle_prompt = generator.generate_image_prompt(
            product_description=args.description,
            brand_style=args.style,
            ad_copy=args.copy
        )
        
        print(f"Claude-optimized DALL-E prompt: {dalle_prompt}")
        
        # Step 2: Generate image with Azure DALL-E
        print("Generating image with Azure DALL-E...")
        image_result = generator.generate_ad_image(
            text_prompt=dalle_prompt,
            image_model=args.model,
            size=args.size,
            quality=args.quality
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
            "claude_optimized_prompt": dalle_prompt,
            "image_generation_result": image_result,
            "image_settings": {
                "model": args.model,
                "size": args.size,
                "quality": args.quality
            }
        }
        
        if args.download:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_filename = f"./output/llm_image_human_copy/li_generated_image_{timestamp}.png"
            saved_path = generator.download_image(image_result['image_url'], image_filename)
            results["downloaded_image"] = saved_path
            print(f"Image downloaded to: {saved_path}")
        
        output_file = args.output or f"./output/llm_image_human_copy/li_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generator.save_results(results, output_file)
        
        print(f"Generated Image URL: {image_result['image_url']}")
        print(f"Claude-Optimized Prompt: {dalle_prompt}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()