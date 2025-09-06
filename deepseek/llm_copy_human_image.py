#!/usr/bin/env python3

import argparse
import sys
import os
from datetime import datetime
from base_generator import DeepSeekHuggingFaceAdGenerator

def main():
    parser = argparse.ArgumentParser(description='DeepSeek V3 does not support image analysis - use text-only generation')
    parser.add_argument('--image', '-i', required=True, help='Path to input image file (for reference only)')
    parser.add_argument('--description', '-d', required=True, help='Describe what you see in the image for text-based generation')
    parser.add_argument('--prompt', '-p', help='Custom prompt for ad copy generation')
    parser.add_argument('--brand-style', '-b', default='modern', help='Brand style')
    parser.add_argument('--target-audience', '-t', default='general', help='Target audience')
    parser.add_argument('--model', '-m', default='DeepSeek-V3-0324', help='DeepSeek model to use')
    parser.add_argument('--output', '-o', help='Output JSON filename')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        sys.exit(1)
    
    os.makedirs("./output/llm_copy_human_image", exist_ok=True)
    
    try:
        generator = DeepSeekHuggingFaceAdGenerator(model_name=args.model)
        
        print("⚠️  Note: DeepSeek V3 does not support vision capabilities.")
        print("Using text-based description for ad copy generation.")
        print(f"Image reference: {args.image}")
        print(f"Product description: {args.description}")
        
        # Use text-based generation instead of image analysis
        result = generator.generate_ad_copy_from_text(
            product_description=args.description,
            brand_style=args.brand_style,
            target_audience=args.target_audience
        )
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
        
        results = {
            "condition": "LC (LLM Copy / Human Image) - Text-based approximation",
            "timestamp": datetime.now().isoformat(),
            "input_image": args.image,
            "product_description": args.description,  # What user described from the image
            "model_used": args.model,
            "brand_style": args.brand_style,
            "target_audience": args.target_audience,
            "custom_prompt": args.prompt,
            "generation_result": result,
            "note": "DeepSeek V3 does not support vision - used text description instead"
        }
        
        output_file = args.output or f"./output/llm_copy_human_image/lc_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generator.save_results(results, output_file)
        
        print("Generated Ad Copy:")
        copy_data = result['copy_data']
        print(f"Primary Copy: {copy_data.get('primary_copy', 'N/A')}")
        print(f"Headline: {copy_data.get('headline', 'N/A')}")
        print(f"Description: {copy_data.get('description', 'N/A')}")
        print(f"CTA: {copy_data.get('cta', 'N/A')}")
        
        print(f"\n💡 Tip: For true vision-based analysis, use Claude or Gemini instead.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()