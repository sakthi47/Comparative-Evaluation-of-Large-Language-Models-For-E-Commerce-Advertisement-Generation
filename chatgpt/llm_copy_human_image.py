#!/usr/bin/env python3

import argparse
import sys
import os
from datetime import datetime
from base_generator import AzureOpenAIImageTextGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate LLM ad copy from human-provided image')
    parser.add_argument('--image', '-i', required=True, help='Path to input image file')
    parser.add_argument('--prompt', '-p', help='Custom prompt for ad copy generation')
    parser.add_argument('--model', '-m', default='gpt-4o-2', help='Azure OpenAI deployment name')
    parser.add_argument('--output', '-o', help='Output JSON filename')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        sys.exit(1)
    
    os.makedirs("./output/llm_copy_human_image", exist_ok=True)
    
    try:
        generator = AzureOpenAIImageTextGenerator()
        
        print(f"Generating ad copy from image: {args.image}")
        
        ad_copy = generator.generate_ad_copy_from_image(
            image_path=args.image,
            custom_prompt=args.prompt,
            deployment_name=args.model
        )
        
        results = {
            "condition": "LC (LLM Copy / Human Image)",
            "timestamp": datetime.now().isoformat(),
            "input_image": args.image,
            "model_used": args.model,
            "generated_copy": ad_copy,
            "custom_prompt": args.prompt
        }
        
        output_file = args.output or f"./output/llm_copy_human_image/lc_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generator.save_results(results, output_file)
        
        print("Generated Ad Copy:")
        print(ad_copy)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()