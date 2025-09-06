#!/usr/bin/env python3

import argparse
import sys
import os
from datetime import datetime
from base_generator import GeminiHuggingFaceAdGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate Gemini ad copy from human-provided image')
    parser.add_argument('--image', '-i', required=True, help='Path to input image file')
    parser.add_argument('--prompt', '-p', help='Custom prompt for ad copy generation')
    parser.add_argument('--model', '-m', default='gemini-1.5-flash', help='Gemini model to use')
    parser.add_argument('--output', '-o', help='Output JSON filename')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        sys.exit(1)
    
    os.makedirs("./output/llm_copy_human_image", exist_ok=True)
    
    try:
        generator = GeminiHuggingFaceAdGenerator(model_name=args.model)
        
        print(f"Generating ad copy from image: {args.image}")
        
        result = generator.generate_ad_copy_from_image(
            image_path=args.image,
            custom_prompt=args.prompt
        )
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
        
        results = {
            "condition": "LC (LLM Copy / Human Image)",
            "timestamp": datetime.now().isoformat(),
            "input_image": args.image,
            "model_used": args.model,
            "custom_prompt": args.prompt,
            "generation_result": result
        }
        
        output_file = args.output or f"./output/llm_copy_human_image/lc_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generator.save_results(results, output_file)
        
        print("Generated Ad Copy:")
        copy_data = result['copy_data']
        print(f"Primary Copy: {copy_data.get('primary_copy', 'N/A')}")
        print(f"Headline: {copy_data.get('headline', 'N/A')}")
        print(f"Description: {copy_data.get('description', 'N/A')}")
        print(f"CTA: {copy_data.get('cta', 'N/A')}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()