try:
    from luma.core.interface.serial import i2c
    from luma.core.render import canvas
    from luma.oled.device import ssd1306
    LUMA_AVAILABLE = True
except ImportError:
    LUMA_AVAILABLE = False

from PIL import Image, ImageDraw, ImageFont
import os
import time
import logging

class DisplayModule:
    """
    Handles OLED interaction.
    Gracefully handles missing hardware/libraries.
    """
    def __init__(self, port=1, address=0x3C, width=128, height=64):
        self.logger = logging.getLogger(__name__)
        self.device = None
        self.width = width
        self.height = height
        
        if LUMA_AVAILABLE:
            try:
                serial = i2c(port=port, address=address)
                self.device = ssd1306(serial, width=width, height=height)
                self.logger.info("OLED Display initialized successfully.")
            except Exception as e:
                self.logger.error(f"Failed to init OLED: {e}")
                self.device = None
        else:
            self.logger.warning("Luma OLED library not found (PC Mode). Display disabled.")

        self.font = ImageFont.load_default() # Use default for now

    def clear(self):
        """Clear display"""
        if self.device:
            self.device.clear()

    def show_text(self, text, size=10):
        """Display text centered"""
        if self.device:
            with canvas(self.device) as draw:
                w, h = draw.textsize(text, font=self.font)
                x = (self.width - w) // 2
                y = (self.height - h) // 2
                draw.text((x, y), text, font=self.font, fill="white")

    def show_face(self, face_image_path):
        """Display face image (resized)"""
        if not self.device or not os.path.exists(face_image_path):
            return

        try:
            img = Image.open(face_image_path).convert('1') # Convert to monochrome
            img = img.resize((self.width, self.height))
            self.device.display(img)
        except Exception as e:
            self.logger.error(f"Failed to display image: {e}")

    def show_emoji(self, emotion):
        """Display emoji based on emotion/calming need"""
        emoji_map = {
            "calm": "assets/emojis/calm.png",
            "happy": "assets/emojis/happy.png",
            "neutral": "assets/emojis/neutral.png",
            "alert": "assets/emojis/alert.png"
        }
        
        path = emoji_map.get(emotion.lower(), emoji_map["neutral"])
        if os.path.exists(path):
            self.show_face(path)
        else:
            self.show_text(f":{emotion}:")
