import os
import anthropic
from openai import AzureOpenAI
from dotenv import load_dotenv
import json
import time
from datetime import datetime
import requests
from typing import Dict, Optional
import base64

load_dotenv()

class ClaudeAzureDALLEAdGenerator:
    def __init__(self):
        self.claude_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        self.azure_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_IMAGE_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-02-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_IMAGE_ENDPOINT")
        )
    
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
                                   custom_prompt: str = None) -> Dict:
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
        
        base64_image = self.encode_image_to_base64(image_path)
        start_time = time.time()
        
        try:
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                temperature=0.7,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": custom_prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            )
            
            generation_time = time.time() - start_time
            copy_content = response.content[0].text
            
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
                                  target_audience: str = "general") -> Dict:
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
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            generation_time = time.time() - start_time
            copy_content = response.content[0].text
            
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
        Create a detailed DALL-E prompt for generating a professional e-commerce ad background image:
        
        PRODUCT: {product_description}
        BRAND STYLE: {brand_style}
        AD COPY CONTEXT: "{ad_copy}"
        
        Requirements for the DALL-E prompt:
        - Create photorealistic, professional quality
        - Include lifestyle context that appeals to target audience
        - Proper lighting and composition for advertising
        - Leave space for text overlay (rule of thirds)
        - High commercial appeal
        - Avoid any text or logos in the image
        
        Generate a single, detailed DALL-E prompt (under 400 characters) that will create an engaging background for this product ad.
        
        Only return the DALL-E prompt, nothing else.
        """
        
        try:
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                temperature=0.6,
                messages=[{"role": "user", "content": prompt}]
            )
            
            dalle_prompt = response.content[0].text.strip()
            dalle_prompt = dalle_prompt.replace('"', '').replace('DALL-E prompt:', '').strip()
            
            return dalle_prompt
            
        except Exception as e:
            return f"Professional {product_description} advertisement background, {brand_style} style, clean composition, commercial photography lighting, space for text overlay"
    
    def generate_ad_image(self, 
                         text_prompt: str,
                         image_model: str = "dall-e-3",
                         size: str = "1024x1024",
                         quality: str = "standard") -> Dict:
        start_time = time.time()
        
        try:
            response = self.azure_client.images.generate(
                model=os.getenv("AZURE_DALLE_DEPLOYMENT_NAME", "dall-e-3"),
                prompt=text_prompt,
                size=size,
                quality=quality,
                n=1
            )
            
            generation_time = time.time() - start_time
            
            return {
                'image_url': response.data[0].url,
                'revised_prompt': response.data[0].revised_prompt if hasattr(response.data[0], 'revised_prompt') else text_prompt,
                'original_prompt': text_prompt,
                'generation_time': generation_time,
                'timestamp': datetime.now().isoformat(),
                'size': size
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'original_prompt': text_prompt,
                'timestamp': datetime.now().isoformat()
            }
    
    def download_image(self, image_url: str, save_path: str) -> str:
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            return save_path
            
        except Exception as e:
            raise Exception(f"Error downloading image: {str(e)}")
    
    def save_results(self, results: dict, filename: str):
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {filename}")
    
    def _parse_copy_fallback(self, copy_content: str) -> Dict:
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