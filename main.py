import threading
from agents.message_pool import MessagePool
from agents.mission_planner import plan_mission
from agents.vision_agent import VisionAgent
from agents.navigator import NavigatorAgent
from agents.guardian import GuardianAgent
import time

# Agent thread wrappers

def mission_planner_thread(operator_prompt, message_pool):
    subgoals = plan_mission(operator_prompt)
    message_pool.post({"type": "subgoals", "data": subgoals})

def vision_agent_thread(image_path, message_pool):
    if image_path:
        vision_agent = VisionAgent()
        vision_context = vision_agent.describe_image(image_path)
        message_pool.post({"type": "vision_context", "data": vision_context})

def navigator_thread(message_pool):
    navigator = NavigatorAgent()
    while True:
        msg = message_pool.get()
        if msg and msg["type"] == "subgoals":
            for step in msg["data"]:
                # Wait for vision context if needed
                vision_context = None
                while True:
                    try:
                        vmsg = message_pool.get(timeout=0.1)
                        if vmsg and vmsg["type"] == "vision_context":
                            vision_context = vmsg["data"]
                            break
                    except:
                        break
                planned_action = navigator.run({"step": step, "vision_context": vision_context})
                message_pool.post({"type": "planned_action", "step": step, "data": planned_action})
        else:
            time.sleep(0.1)

def guardian_thread(message_pool):
    guardian = GuardianAgent()
    while True:
        msg = message_pool.get()
        if msg and msg["type"] == "planned_action":
            validation = guardian.validate(msg["step"], msg["data"])
            message_pool.post({"type": "guardian_validation", "step": msg["step"], "data": validation})
        else:
            time.sleep(0.1)

def run_mission(operator_prompt, image_path=None):
    print("\n== Start: Concurrent Mission ==")
    message_pool = MessagePool()

    threads = [
        threading.Thread(target=mission_planner_thread, args=(operator_prompt, message_pool)),
        threading.Thread(target=vision_agent_thread, args=(image_path, message_pool)),
        threading.Thread(target=navigator_thread, args=(message_pool,)),
        threading.Thread(target=guardian_thread, args=(message_pool,)),
    ]
    for t in threads:
        t.daemon = True
        t.start()

    # Monitor guardian validation and print results
    while True:
        msg = message_pool.get()
        if msg and msg["type"] == "guardian_validation":
            print(f"Guardian validation for step '{msg['step']}': {msg['data']}")
            if msg["data"] != "OK":
                print("Guardian rejected action. Stopping mission.")
                break
        time.sleep(0.1)

if __name__ == "__main__":
    # Example usage:
    run_mission("Leć do przodu i wyląduj.", image_path="person_img.jpeg")
