#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime
import os
from base_generator import AzureOpenAIImageTextGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate both LLM ad copy and LLM image')
    parser.add_argument('--description', '-d', required=True, help='Product description')
    parser.add_argument('--brand-style', '-b', default='modern', help='Brand style')
    parser.add_argument('--target-audience', '-t', default='general', help='Target audience')
    parser.add_argument('--text-model', default='gpt-4o-2', help='Text generation model deployment name')
    parser.add_argument('--image-model', default='dall-e-3', help='Image generation model deployment name')
    parser.add_argument('--size', default='1024x1024', help='Image size')
    parser.add_argument('--quality', default='standard', help='Image quality')
    parser.add_argument('--output', '-o', help='Output JSON filename')
    parser.add_argument('--download', action='store_true', help='Download generated image')
    
    args = parser.parse_args()
    
    try:
        os.makedirs("./output/llm_both_generated", exist_ok=True)
        
        generator = AzureOpenAIImageTextGenerator()
        
        print(f"Generating complete ad for: {args.description}")
        
        print("Step 1: Generating ad copy...")
        ad_copy = generator.generate_ad_copy_from_text(
            product_description=args.description,
            brand_style=args.brand_style,
            target_audience=args.target_audience,
            deployment_name=args.text_model
        )
        
        print("Step 2: Generating product image...")
        image_prompt = f"""
        Professional product photography of {args.description}, 
        {args.brand_style} style, high quality, clean background, 
        commercial advertising photo, well-lit, appealing to {args.target_audience},
        suitable for Facebook/Instagram ads, no text or logos
        """
        
        image_url = generator.generate_ad_image(
            text_prompt=image_prompt,
            image_model=args.image_model,
            size=args.size,
            quality=args.quality
        )
        
        results = {
            "condition": "LCI (LLM Copy & LLM Image)",
            "timestamp": datetime.now().isoformat(),
            "product_description": args.description,
            "brand_style": args.brand_style,
            "target_audience": args.target_audience,
            "generated_copy": ad_copy,
            "generated_image_url": image_url,
            "image_prompt": image_prompt,
            "models_used": {
                "text_model": args.text_model,
                "image_model": args.image_model
            },
            "image_settings": {
                "size": args.size,
                "quality": args.quality
            }
        }
        
        if args.download:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_filename = f"./output/llm_both_generated/lci_generated_image_{timestamp}.png"
            saved_path = generator.download_image(image_url, image_filename)
            results["downloaded_image"] = saved_path
            print(f"Image downloaded to: {saved_path}")
        
        output_file = args.output or f"./output/llm_both_generated/lci_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generator.save_results(results, output_file)
        
        print("\nGenerated Ad Copy:")
        print(ad_copy)
        print(f"\nGenerated Image URL: {image_url}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()