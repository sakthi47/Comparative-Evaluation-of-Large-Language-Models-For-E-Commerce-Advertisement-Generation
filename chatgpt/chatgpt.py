import os
import base64
import requests
from openai import AzureOpenAI
from typing import Optional
from dotenv import load_dotenv
import json

load_dotenv()

class AzureOpenAIImageTextGenerator:
    def __init__(self, 
                 text_api_key: str, 
                 text_endpoint: str, 
                 image_api_key: str = None,
                 image_endpoint: str = None,
                 api_version: str = "2024-02-01"):
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
        
        self.client = self.text_client
    
    def encode_image_to_base64(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        except Exception as e:
            raise Exception(f"Error encoding image: {str(e)}")
    
    def generate_ad_copy(self, 
                        image_path: str, 
                        prompt: str = None,
                        deployment_name: str = "gpt-4o-2",
                        max_tokens: int = 300,
                        temperature: float = 0.7) -> str:
        if prompt is None:
            prompt = """
            You are an expert copywriter for e-commerce advertising. 
            Analyze this product image and create compelling advertising copy that includes:
            
            1. A catchy headline (max 25 words)
            2. Product description highlighting key features
            3. Call-to-action that drives conversions
            4. Target emotional triggers for potential buyers
            
            Keep the tone persuasive, clear, and suitable for Facebook/Instagram ads.
            Focus on benefits over features.
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
                                "text": prompt
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
                max_tokens=max_tokens,
                temperature=temperature
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
    
    def download_image(self, image_url: str, save_path: str) -> str:
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            return save_path
            
        except Exception as e:
            raise Exception(f"Error downloading image: {str(e)}")
    
    def create_complete_ad(self, 
                          product_description: str,
                          brand_style: str = "modern",
                          target_audience: str = "general",
                          deployment_name: str = "gpt-4o-2") -> dict:
        copy_prompt = f"""
        Create compelling advertising copy for a {product_description}.
        Brand style: {brand_style}
        Target audience: {target_audience}
        
        Include:
        1. Catchy headline
        2. Product benefits
        3. Strong call-to-action
        4. Keep it suitable for social media ads
        """
        
        try:
            copy_response = self.text_client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "user", "content": copy_prompt}],
                max_tokens=300,
                temperature=0.7
            )
            
            ad_copy = copy_response.choices[0].message.content
            
            image_prompt = f"""
            Professional product photography of {product_description}, 
            {brand_style} style, high quality, clean background, 
            commercial advertising photo, well-lit, appealing to {target_audience},
            suitable for Facebook/Instagram ads
            """
            
            image_url = self.generate_ad_image(image_prompt)
            
            return {
                "ad_copy": ad_copy,
                "image_url": image_url,
                "image_prompt": image_prompt
            }
            
        except Exception as e:
            raise Exception(f"Error creating complete ad: {str(e)}")

def main():
    TEXT_API_KEY = os.getenv("AZURE_OPENAI_TEXT_KEY")
    TEXT_ENDPOINT = os.getenv("AZURE_OPENAI_TEXT_ENDPOINT")
    
    IMAGE_API_KEY = os.getenv("AZURE_OPENAI_IMAGE_KEY")
    IMAGE_ENDPOINT = os.getenv("AZURE_OPENAI_IMAGE_ENDPOINT")
    
    if not TEXT_API_KEY:
        TEXT_API_KEY = "your_text_api_key_here"
        TEXT_ENDPOINT = "https://your-text-resource.openai.azure.com/"
    
    if not IMAGE_API_KEY:
        IMAGE_API_KEY = "your_image_api_key_here"  
        IMAGE_ENDPOINT = "https://your-image-resource.openai.azure.com/"
    
    IMAGE_PATH = "bottle.jpeg"
    
    TESTING_MODE = True
    
    TEST_TYPE = "complete_ad"
    
    try:
        generator = AzureOpenAIImageTextGenerator(
            text_api_key=TEXT_API_KEY,
            text_endpoint=TEXT_ENDPOINT,
            image_api_key=IMAGE_API_KEY,
            image_endpoint=IMAGE_ENDPOINT
        )
        
        if TEST_TYPE == "image_to_text":
            ad_copy = generator.generate_ad_copy(
                image_path=IMAGE_PATH,
                deployment_name="gpt-4o-2",
                max_tokens=300,
                temperature=0.7
            )
            
        elif TEST_TYPE == "text_to_image":
            product_description = "premium water bottle, sleek design, eco-friendly"
            image_prompt = f"Professional product photo of {product_description}, modern style, clean white background, high quality commercial photography"
            
            image_url = generator.generate_ad_image(
                text_prompt=image_prompt,
                size="1024x1024",
                quality="standard"
            )
            
            downloaded_path = generator.download_image(image_url, "generated_ad_image.png")
            
        elif TEST_TYPE == "complete_ad":
            complete_ad = generator.create_complete_ad(
                product_description="premium eco-friendly water bottle",
                brand_style="modern minimalist",
                target_audience="health-conscious young adults",
                deployment_name="gpt-4o-2"
            )
            
            downloaded_path = generator.download_image(complete_ad["image_url"], "complete_ad_image.png")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check your image path and make sure the file exists.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your API credentials and deployment names.")

def test_experimental_conditions():
    generator = AzureOpenAIImageTextGenerator(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    conditions = {
        "H": {
            "name": "Human Baseline",
            "description": "Use existing bottle.jpeg + manual copy"
        },
        "LC": {
            "name": "LLM Copy / Human Image", 
            "test": "image_to_text",
            "image_path": "bottle.jpeg"
        },
        "LI": {
            "name": "LLM Image / Human Copy",
            "test": "text_to_image", 
            "prompt": "premium water bottle, eco-friendly, sleek design"
        },
        "LCI": {
            "name": "LLM Copy & LLM Image",
            "test": "complete_ad",
            "product": "premium eco-friendly water bottle"
        },
        "E": {
            "name": "Multi-Model Ensemble",
            "description": "Best-of-N selection across multiple LLMs"
        }
    }
    
    for condition_code, config in conditions.items():
        if condition_code == "H":
            print("👤 Human baseline - manual process")
            
        elif condition_code == "LC":
            ad_copy = generator.generate_ad_copy(
                image_path=config["image_path"],
                max_tokens=200,
                temperature=0.7
            )
            print("Generated Copy:", ad_copy[:100] + "...")
            
        elif condition_code == "LI":
            image_url = generator.generate_ad_image(config["prompt"])
            print("Generated Image URL:", image_url)
            
        elif condition_code == "LCI":
            complete_ad = generator.create_complete_ad(
                product_description=config["product"],
                brand_style="modern",
                target_audience="health-conscious consumers"
            )
            print("Generated Copy:", complete_ad["ad_copy"][:100] + "...")
            print("Generated Image:", complete_ad["image_url"])
            
        elif condition_code == "E":
            print("🤖 Ensemble method - combine multiple LLM outputs")

def quick_image_test():
    generator = AzureOpenAIImageTextGenerator(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    try:
        prompt = "modern water bottle, minimalist design, white background, product photography"
        
        image_url = generator.generate_ad_image(prompt, size="1024x1024")
        
        saved_path = generator.download_image(image_url, "test_generated_bottle.png")
        
    except Exception as e:
        print(f" Error: {e}")

def run_lci_experiment():
    generator = AzureOpenAIImageTextGenerator(
        text_api_key=os.getenv("AZURE_OPENAI_TEXT_KEY"),
        text_endpoint=os.getenv("AZURE_OPENAI_TEXT_ENDPOINT"),
        image_api_key=os.getenv("AZURE_OPENAI_IMAGE_KEY"),
        image_endpoint=os.getenv("AZURE_OPENAI_IMAGE_ENDPOINT")
    )
    
    product_info = {
        "name": "Premium Eco Water Bottle",
        "description": "sustainable stainless steel water bottle with temperature control",
        "target_audience": "health-conscious millennials",
        "brand_style": "modern minimalist",
        "key_features": ["BPA-free", "24hr temperature retention", "leak-proof", "eco-friendly"]
    }
    
    try:
        image_prompt = f"""
        Professional product photography of {product_info['description']}, 
        {product_info['brand_style']} design, clean white background, 
        studio lighting, commercial quality, e-commerce product shot,
        sleek and modern appearance
        """
        
        image_url = generator.generate_ad_image(
            text_prompt=image_prompt,
            image_model="dall-e-3",
            size="1024x1024",
            quality="standard",
            style="vivid"
        )
        
        image_filename = "lci_generated_product.png"
        saved_image_path = generator.download_image(image_url, image_filename)
        
        copy_prompt = f"""
        Create compelling Facebook/Instagram ad copy for {product_info['name']}.
        
        Product details:
        - Description: {product_info['description']}
        - Key features: {', '.join(product_info['key_features'])}
        - Target audience: {product_info['target_audience']}
        - Brand style: {product_info['brand_style']}
        
        Requirements:
        1. Catchy headline (max 25 words)
        2. Highlight key benefits (not just features)
        3. Strong call-to-action
        4. Emotional appeal for target audience
        5. Suitable for social media (casual but professional tone)
        
        Format: Headline, body text, call-to-action
        """
        
        copy_response = generator.text_client.chat.completions.create(
            model="gpt-4o-2",
            messages=[{"role": "user", "content": copy_prompt}],
            max_tokens=300,
            temperature=0.7
        )
        
        ad_copy = copy_response.choices[0].message.content
        
        results = {
            "condition": "LCI (LLM Copy & LLM Image)",
            "product_info": product_info,
            "generated_image_url": image_url,
            "generated_image_path": saved_image_path,
            "generated_ad_copy": ad_copy,
            "image_prompt": image_prompt.strip(),
            "copy_prompt": copy_prompt.strip()
        }
        
        import json
        with open("lci_experiment_result.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        print(f"Error in LCI experiment: {e}")
        return None

if __name__ == "__main__":
    run_lci_experiment()