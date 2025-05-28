import threading
from agents.message_pool import MessagePool
from agents.mission_planner import MissionPlannerAgent
from agents.vision_agent import VisionAgent
from agents.navigator import NavigatorAgent
from agents.guardian import GuardianAgent
import time
from dotenv import load_dotenv

load_dotenv()


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

