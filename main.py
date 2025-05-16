import logging
from pathlib import Path
from core.system_integrator import SystemIntegrator
from utils.config_manager import ConfigManager
from ui.menu_handler import MenuHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Path("logs/super_ai.log")),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class SuperAI:
    def __init__(self):
        self.config = ConfigManager()
        self.integrator = SystemIntegrator()
        self.menu = MenuHandler()
        logger.info("SuperAI system initialized")

    def run(self):
        try:
            while True:
                choice = self.menu.display_main_menu()
                if choice == "betting":
                    sport = self.menu.display_sports_menu()
                    if sport:
                        predictions = self.integrator.get_betting_predictions(sport)
                        self.menu.display_predictions(predictions)
                elif choice == "forex":
                    pair = self.menu.display_forex_menu()
                    if pair:
                        predictions = self.integrator.get_forex_predictions(pair)
                        self.menu.display_predictions(predictions)
                elif choice == "quit":
                    break
        except Exception as e:
            logger.error(f"System error: {str(e)}")
            raise


def main():
    try:
        ai = SuperAI()
        ai.run()
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())

# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------
