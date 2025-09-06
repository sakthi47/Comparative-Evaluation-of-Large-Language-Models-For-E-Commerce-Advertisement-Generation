#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime
import os
from base_generator import ClaudeAzureDALLEAdGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate both Claude ad copy and Azure DALL-E image')
    parser.add_argument('--description', '-d', required=True, help='Product description')
    parser.add_argument('--brand-style', '-b', default='modern', help='Brand style')
    parser.add_argument('--target-audience', '-t', default='general', help='Target audience')
    parser.add_argument('--image-model', default='dall-e-3', help='Image generation model deployment name')
    parser.add_argument('--size', default='1024x1024', help='Image size')
    parser.add_argument('--quality', default='standard', help='Image quality')
    parser.add_argument('--output', '-o', help='Output JSON filename')
    parser.add_argument('--download', action='store_true', help='Download generated image')
    
    args = parser.parse_args()
    
    try:
        os.makedirs("./output/llm_both_generated", exist_ok=True)
        
        generator = ClaudeAzureDALLEAdGenerator()
        
        print(f"Generating complete ad for: {args.description}")
        
        # Step 1: Generate ad copy with Claude
        print("Step 1: Generating ad copy with Claude...")
        copy_result = generator.generate_ad_copy_from_text(
            product_description=args.description,
            brand_style=args.brand_style,
            target_audience=args.target_audience
        )
        
        if 'error' in copy_result:
            print(f"Error generating copy: {copy_result['error']}")
            sys.exit(1)
        
        primary_copy = copy_result['copy_data'].get('primary_copy', '')
        
        # Step 2: Generate optimized image prompt with Claude
        print("Step 2: Generating optimized DALL-E prompt with Claude...")
        dalle_prompt = generator.generate_image_prompt(
            product_description=args.description,
            brand_style=args.brand_style,
            ad_copy=primary_copy
        )
        
        # Step 3: Generate image with Azure DALL-E
        print("Step 3: Generating image with Azure DALL-E...")
        image_result = generator.generate_ad_image(
            text_prompt=dalle_prompt,
            image_model=args.image_model,
            size=args.size,
            quality=args.quality
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
            "copy_generation_result": copy_result,
            "claude_optimized_prompt": dalle_prompt,
            "image_generation_result": image_result,
            "image_settings": {
                "model": args.image_model,
                "size": args.size,
                "quality": args.quality
            }
        }
        
        if args.download:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_filename = f"./output/llm_both_generated/lci_generated_image_{timestamp}.png"
            saved_path = generator.download_image(image_result['image_url'], image_filename)
            results["downloaded_image"] = saved_path
            print(f"Image downloaded to: {saved_path}")
        
        output_file = args.output or f"./output/llm_both_generated/lci_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generator.save_results(results, output_file)
        
        print("\nGenerated Ad Copy:")
        copy_data = copy_result['copy_data']
        print(f"Primary Copy: {copy_data.get('primary_copy', 'N/A')}")
        print(f"Headline: {copy_data.get('headline', 'N/A')}")
        print(f"Description: {copy_data.get('description', 'N/A')}")
        print(f"CTA: {copy_data.get('cta', 'N/A')}")
        
        print(f"\nGenerated Image URL: {image_result['image_url']}")
        print(f"Claude-Optimized DALL-E Prompt: {dalle_prompt}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()