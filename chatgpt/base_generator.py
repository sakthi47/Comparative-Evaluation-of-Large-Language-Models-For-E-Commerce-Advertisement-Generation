import os
import base64
import requests
from openai import AzureOpenAI
from dotenv import load_dotenv
import json

load_dotenv()

class AzureOpenAIImageTextGenerator:
    def __init__(self, 
                 text_api_key: str = None, 
                 text_endpoint: str = None, 
                 image_api_key: str = None,
                 image_endpoint: str = None,
                 api_version: str = "2024-02-01"):
        
        text_api_key = text_api_key or os.getenv("AZURE_OPENAI_TEXT_KEY")
        text_endpoint = text_endpoint or os.getenv("AZURE_OPENAI_TEXT_ENDPOINT")
        image_api_key = image_api_key or os.getenv("AZURE_OPENAI_IMAGE_KEY")
        image_endpoint = image_endpoint or os.getenv("AZURE_OPENAI_IMAGE_ENDPOINT")
        
        self.text_client = AzureOpenAI(
            api_key=text_api_key,
            api_version=api_version,
            azure_endpoint=text_endpoint
        )
        
        self.image_client = AzureOpenAI(
            api_key=image_api_key or text_api_key,
            api_version=api_version,
            azure_endpoint=image_endpoint or text_endpoint
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
                                   custom_prompt: str = None,
                                   deployment_name: str = "gpt-4o-2") -> str:
        if custom_prompt is None:
            custom_prompt = """
            You are an expert copywriter for e-commerce advertising. 
            Analyze this product image and create compelling advertising copy that includes:
            
            1. A catchy headline (max 25 words)
            2. Product description highlighting key features
            3. Call-to-action that drives conversions
            4. Target emotional triggers for potential buyers
            
            Keep the tone persuasive, clear, and suitable for Facebook/Instagram ads.
            Focus on benefits over features.
            
            Format as JSON with keys: headline, description, cta
            """
        
        base64_image = self.encode_image_to_base64(image_path)
        
        try:
            response = self.text_client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": custom_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Error generating text: {str(e)}")
    
    def generate_ad_image(self, 
                         text_prompt: str,
                         image_model: str = "dall-e-3",
                         size: str = "1024x1024",
                         quality: str = "standard",
                         style: str = "vivid") -> str:
        try:
            response = self.image_client.images.generate(
                model=image_model,
                prompt=text_prompt,
                size=size,
                quality=quality,
                style=style,
                n=1
            )
            
            return response.data[0].url
            
        except Exception as e:
            raise Exception(f"Error generating image: {str(e)}")
    
    def generate_ad_copy_from_text(self,
                                  product_description: str,
                                  brand_style: str = "modern",
                                  target_audience: str = "general",
                                  deployment_name: str = "gpt-4o-2") -> str:
        copy_prompt = f"""
        Create compelling advertising copy for a {product_description}.
        Brand style: {brand_style}
        Target audience: {target_audience}
        
        Include:
        1. Catchy headline (max 25 words)
        2. Product description highlighting key benefits
        3. Strong call-to-action
        4. Keep it suitable for social media ads
        
        Format as JSON with keys: headline, description, cta
        """
        
        try:
            copy_response = self.text_client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "user", "content": copy_prompt}],
                max_tokens=300,
                temperature=0.7
            )
            
            return copy_response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Error generating copy: {str(e)}")
    
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