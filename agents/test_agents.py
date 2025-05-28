from agents.mission_planner import plan_mission
from agents.vision_agent import describe_image
from agents.navigator import NavigatorAgent
from agents.guardian import GuardianAgent


print("--- Mission Planner Test ---")
mission = plan_mission("Wystartuj, leć 10m na północ, zrób zdjęcie, wróć i wyląduj.")
print(mission)

print("\n--- Vision Agent Test ---")
try:
    vision = describe_image("frame.jpg")
    print(vision)
except Exception as e:
    print("Vision agent error:", e)

print("\n--- Navigator + Guardian Test ---")
navigator = NavigatorAgent()
guardian = GuardianAgent()
for step in mission if isinstance(mission, list) else []:
    action = navigator.run(str(step))
    print(f"Step: {step}\nProposed action: {action}")
    validation = guardian.validate(step, action)
    print(f"Guardian validation: {validation}")
    if validation != "OK":
        print("Guardian rejected the action. Stopping test.")
        break
