from agents.mission_planner import plan_mission
from agents.vision_agent import describe_image
from agents.navigator import NavigatorAgent
from agents.guardian import GuardianAgent
from langchain.schema import HumanMessage

def run_mission(operator_prompt, image_path=None):
    print("\n== Etap 1: Planowanie ==")
    subgoals = plan_mission(operator_prompt)
    print(f"Podcele: {subgoals}")

    vision_context = None
    if image_path:
        print("\n== Etap 2: Opis wizji ==")
        vision_context = describe_image(image_path)
        print(f"Wizja: {vision_context}")

    print("\n== Etap 3: Nawigacja + Guardian ==")
    navigator = NavigatorAgent()
    guardian = GuardianAgent()
    for step in subgoals:
        planned_action = navigator.run({"step": step, "vision_context": vision_context})
        print(f"\nProponowana akcja: {planned_action}")
        validation = guardian.validate(step, planned_action)
        if validation == "OK":
            print("Akcja zaakceptowana przez Guardian.")
        else:
            print(f"Guardian odrzucił akcję: {validation}")
            break

if __name__ == "__main__":
    # Example usage:
    run_mission("Leć do przodu i wyląduj.", image_path="person_img.jpeg")
