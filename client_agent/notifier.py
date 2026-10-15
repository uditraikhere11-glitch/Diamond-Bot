import logging
import aiohttp

logger = logging.getLogger(__name__)

class Notifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.enabled = bool(token and chat_id)

    async def send(self, message: str):
        logger.info(f"NOTIFY: {message}")
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(f"Failed to send Telegram notification: {resp.status} {body}")
        except Exception as e:
            logger.error(f"Exception sending Telegram notification: {e}")
