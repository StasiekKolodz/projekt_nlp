import threading
from agents.message_pool import MessagePool
from agents.mission_planner import MissionPlannerAgent
from agents.vision_agent import VisionAgent
from agents.navigator import NavigatorAgent
from agents.guardian import GuardianAgent
import time
from dotenv import load_dotenv
import logging
import warnings

load_dotenv()

logging.getLogger("langchain_core").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)
logging.getLogger("langgraph").setLevel(logging.ERROR)
logging.getLogger("langchain_openai").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    # run_mission("Leć do przodu i wyląduj.", image_path="person_img.jpeg")
    message_pool = MessagePool()
    
    guardian_agent = GuardianAgent(message_pool)
    guardian_agent.start()
    navigator_agent = NavigatorAgent(message_pool)
    navigator_agent.start()
    vision_agent = VisionAgent(message_pool)
    vision_agent.start()
    
    time.sleep(1)
    mp = MissionPlannerAgent(message_pool)
    mp.run()

