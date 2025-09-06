#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime
import os
from base_generator import AzureOpenAIImageTextGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate LLM image from human-provided description')
    parser.add_argument('--description', '-d', required=True, help='Product description for image generation')
    parser.add_argument('--style', '-s', default='modern minimalist', help='Brand/visual style')
    parser.add_argument('--size', default='1024x1024', help='Image size (1024x1024, 1024x1792, 1792x1024)')
    parser.add_argument('--quality', default='standard', help='Image quality (standard, hd)')
    parser.add_argument('--model', '-m', default='dall-e-3', help='DALL-E model deployment name')
    parser.add_argument('--output', '-o', help='Output JSON filename')
    parser.add_argument('--download', action='store_true', help='Download generated image')
    
    args = parser.parse_args()
    
    try:
        os.makedirs("./output/llm_image_human_copy", exist_ok=True)
        
        generator = AzureOpenAIImageTextGenerator()
        
        print(f"Generating image for: {args.description}")
        
        image_prompt = f"""
        Professional product photography of {args.description}, 
        {args.style} style, high quality, clean background, 
        commercial advertising photo, well-lit, suitable for Facebook/Instagram ads,
        no text or logos in image
        """
        
        image_url = generator.generate_ad_image(
            text_prompt=image_prompt,
            image_model=args.model,
            size=args.size,
            quality=args.quality
        )
        
        results = {
            "condition": "LI (LLM Image / Human Copy)",
            "timestamp": datetime.now().isoformat(),
            "product_description": args.description,
            "brand_style": args.style,
            "image_prompt": image_prompt,
            "generated_image_url": image_url,
            "image_settings": {
                "model": args.model,
                "size": args.size,
                "quality": args.quality
            }
        }
        
        if args.download:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_filename = f"./output/llm_image_human_copy/li_generated_image_{timestamp}.png"
            saved_path = generator.download_image(image_url, image_filename)
            results["downloaded_image"] = saved_path
            print(f"Image downloaded to: {saved_path}")
        
        output_file = args.output or f"./output/llm_image_human_copy/li_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generator.save_results(results, output_file)
        
        print(f"Generated Image URL: {image_url}")
        print(f"Image Prompt Used: {image_prompt}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()