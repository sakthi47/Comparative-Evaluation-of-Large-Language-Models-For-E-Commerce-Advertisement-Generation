import os
import base64
import requests
import google.generativeai as genai
from typing import Optional
from dotenv import load_dotenv
import json
from PIL import Image
import io
from huggingface_hub import InferenceClient
import time
from datetime import datetime

load_dotenv()

class HuggingFaceImageGenerator:
    def __init__(self, hf_api_key: str):
        self.client = InferenceClient(token=hf_api_key)
        self.image_models = {
            "flux-schnell": "black-forest-labs/FLUX.1-schnell",
            "stable-diffusion": "stabilityai/stable-diffusion-xl-base-1.0",
            "playground": "playgroundai/playground-v2.5-1024px-aesthetic",
        }
    
    def generate_product_image(self, 
                             prompt: str, 
                             model: str = "flux-schnell",
                             width: int = 1024, 
                             height: int = 1024,
                             num_inference_steps: int = 4) -> Image.Image:
        try:
            model_id = self.image_models.get(model, self.image_models["flux-schnell"])
            
            enhanced_prompt = f"""
            {prompt}, professional product photography, high quality, clean white background, 
            commercial advertising photo, well-lit, sharp focus, centered composition, 
            suitable for e-commerce, marketing materials, 8k resolution, photorealistic
            """
            
            image = self.client.text_to_image(
                prompt=enhanced_prompt,
                model=model_id,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps
            )
            
            return image
            
        except Exception as e:
            if model != "stable-diffusion":
                return self.generate_product_image(prompt, "stable-diffusion", width, height, 8)
            raise Exception(f"Image generation failed: {str(e)}")
    
    def save_image(self, image: Image.Image, filename: str, quality: int = 95) -> str:
        try:
            if not filename.endswith(('.jpg', '.jpeg', '.png')):
                filename += '.jpg'
            
            image.save(filename, 'JPEG', quality=quality, optimize=True)
            return filename
            
        except Exception as e:
            raise Exception(f"Error saving image: {str(e)}")

class GeminiHuggingFaceAdGenerator:
    def __init__(self, 
                 google_api_key: str = None,
                 hf_api_key: str = None,
                 model_name: str = "gemini-1.5-flash"):
        
        google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        hf_api_key = hf_api_key or os.getenv("HF_TOKEN")
        
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=300,
            top_p=0.8,
            top_k=40
        )
        
        self.image_generator = HuggingFaceImageGenerator(hf_api_key)
    
    def encode_image_to_base64(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        except Exception as e:
            raise Exception(f"Error encoding image: {str(e)}")
    
    def generate_ad_copy_from_image(self, 
                                   image_path: str, 
                                   custom_prompt: str = None) -> dict:
        if custom_prompt is None:
            custom_prompt = """
            You are an expert copywriter for e-commerce advertising. 
            Analyze this product image and create compelling advertising copy.
            
            Please provide the single best version only:
            1. Primary ad copy (under 125 chars)
            2. Headline (under 40 chars)
            3. Description (under 30 words)
            4. Call-to-action button text
            
            Format as JSON with these exact keys: primary_copy, headline, description, cta
            """
        
        start_time = time.time()
        
        try:
            image = Image.open(image_path)
            
            response = self.model.generate_content(
                [custom_prompt, image],
                generation_config=self.generation_config
            )
            
            generation_time = time.time() - start_time
            copy_content = response.text
            
            try:
                parsed_response = json.loads(copy_content)
            except json.JSONDecodeError:
                parsed_response = self._parse_copy_fallback(copy_content)
            
            return {
                'copy_data': parsed_response,
                'raw_response': copy_content,
                'generation_time': generation_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_ad_copy_from_text(self,
                                  product_description: str,
                                  brand_style: str = "modern",
                                  target_audience: str = "general") -> dict:
        prompt = f"""
        Create compelling advertising copy for a {product_description}.
        Brand style: {brand_style}
        Target audience: {target_audience}
        
        Requirements:
        - Keep primary copy under 125 characters
        - Include compelling call-to-action
        - Create urgency or emotional appeal
        - Highlight key benefits, not just features
        
        Please provide the single best version only:
        1. Primary ad copy (under 125 chars)
        2. Headline (under 40 chars)
        3. Description (under 30 words)
        4. Call-to-action button text
        
        Format as JSON with these exact keys: primary_copy, headline, description, cta
        """
        
        start_time = time.time()
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            generation_time = time.time() - start_time
            copy_content = response.text
            
            try:
                parsed_response = json.loads(copy_content)
            except json.JSONDecodeError:
                parsed_response = self._parse_copy_fallback(copy_content)
            
            return {
                'copy_data': parsed_response,
                'raw_response': copy_content,
                'generation_time': generation_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_image_prompt(self, product_description: str, brand_style: str, ad_copy: str = "") -> str:
        prompt = f"""
        Create a detailed image generation prompt for a product photography of {product_description}.
        
        Brand style: {brand_style}
        Ad copy context: "{ad_copy}"
        
        The prompt should describe:
        1. Product appearance and materials
        2. Lighting and photography style
        3. Background and composition
        4. Colors and mood
        5. Any props or lifestyle elements
        
        Keep it concise but detailed for AI image generation. Focus on visual elements only.
        Only return the image prompt, nothing else.
        """
        
        try:
            response = self.model.generate_content(
                prompt, 
                generation_config=self.generation_config
            )
            return response.text.strip()
        except Exception as e:
            return f"{product_description}, {brand_style} style, professional product photography, clean background, high quality"
    
    def generate_ad_image(self, 
                         text_prompt: str,
                         image_model: str = "flux-schnell",
                         width: int = 1024,
                         height: int = 1024) -> dict:
        start_time = time.time()
        
        try:
            image = self.image_generator.generate_product_image(
                prompt=text_prompt,
                model=image_model,
                width=width,
                height=height
            )
            
            generation_time = time.time() - start_time
            
            # Convert to standard PIL Image if it's a different format
            if hasattr(image, 'convert'):
                image = image.convert('RGB')
            
            return {
                'image_object': image,
                'original_prompt': text_prompt,
                'generation_time': generation_time,
                'timestamp': datetime.now().isoformat(),
                'model': image_model,
                'size': f"{width}x{height}",
                'image_format': str(type(image).__name__)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'original_prompt': text_prompt,
                'timestamp': datetime.now().isoformat()
            }
    
    def download_image(self, image: Image.Image, save_path: str) -> str:
        try:
            return self.image_generator.save_image(image, save_path)
        except Exception as e:
            raise Exception(f"Error saving image: {str(e)}")
    
    def save_results(self, results: dict, filename: str):
        results_copy = results.copy()
        
        # Handle nested image objects in the results
        def clean_image_objects(obj):
            if isinstance(obj, dict):
                cleaned = {}
                for key, value in obj.items():
                    if key == 'image_object':
                        cleaned[key] = f"<PIL.Image object - not serializable>"
                    else:
                        cleaned[key] = clean_image_objects(value)
                return cleaned
            elif isinstance(obj, list):
                return [clean_image_objects(item) for item in obj]
            else:
                return obj
        
        results_copy = clean_image_objects(results_copy)
        
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        print(f"Results saved to: {filename}")
    
    def _parse_copy_fallback(self, copy_content: str) -> dict:
        lines = copy_content.split('\n')
        result = {
            'primary_copy': '',
            'headline': '',
            'description': '',
            'cta': ''
        }
        
        for line in lines:
            if 'primary' in line.lower() or '1.' in line:
                result['primary_copy'] = line.split(':', 1)[-1].strip()
            elif 'headline' in line.lower() or '2.' in line:
                result['headline'] = line.split(':', 1)[-1].strip()
            elif 'description' in line.lower() or '3.' in line:
                result['description'] = line.split(':', 1)[-1].strip()
            elif 'call-to-action' in line.lower() or 'cta' in line.lower() or '4.' in line:
                result['cta'] = line.split(':', 1)[-1].strip()
        
        return result