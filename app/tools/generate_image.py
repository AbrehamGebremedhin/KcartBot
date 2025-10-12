from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from app.core.config import get_settings
from app.tools.base import ToolBase
from typing import Any, Dict

class ImageGenerator(ToolBase):
	"""
	Tool to generate images using Gemini API.
	"""
	def __init__(self):
		super().__init__(
			name="ImageGenerator",
			description="Generates images for fruits, vegetables, and dairy products using the Gemini API."
		)

	async def run(self, subject: str,context: Dict[str, Any] = None) -> str:
		settings = get_settings()
		client = genai.Client(api_key=settings.gemini_api_key)

		prompt = (
			f"Generate an image for {subject}, there should only be {subject} in the frame. The {subject} should be shown as packed and fresh on the package add a tag KCartBot in bold and powered by ChipChip"
		)

		response = client.models.generate_content(
			model="gemini-2.5-flash-image",
			contents=[prompt],
		)

		import os
		import re
		image_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'images')
		os.makedirs(image_dir, exist_ok=True)
		# Sanitize subject for filename
		safe_subject = re.sub(r'[^a-zA-Z0-9_-]', '_', subject.strip())
		image_path = os.path.join(image_dir, f"{safe_subject}.png")
		# Delete existing image if present
		if os.path.exists(image_path):
			os.remove(image_path)
		saved = False
		for part in response.candidates[0].content.parts:
			if part.inline_data is not None:
				image = Image.open(BytesIO(part.inline_data.data))
				image.save(image_path)
				saved = True
				break  # Only save the first image

		if saved:
			return f"Image generated and saved: {image_path}"
		else:
			return "No image was generated."

