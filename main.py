from agents.mission_planner import plan_mission
from agents.vision_agent import VisionAgent
from agents.navigator import NavigatorAgent
from agents.guardian import GuardianAgent
from agents.reflection import ReflectionAgent
from langchain.schema import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def run_mission(operator_prompt, image_path=None):
    print("\n== Etap 1: Planowanie ==")
    subgoals = plan_mission(operator_prompt)
    print(f"Podcele: {subgoals}")

    vision_context = None
    if image_path:
        print("\n== Etap 2: Opis wizji ==")
        vision_agent = VisionAgent()
        vision_context = vision_agent.describe_image(image_path)
        print(f"Wizja: {vision_context}")

    print("\n== Etap 3: Nawigacja + Guardian ==")
    navigator = NavigatorAgent()
    guardian = GuardianAgent()
    reflection_agent = ReflectionAgent()
    for step in subgoals:
        planned_action = navigator.run({"step": step, "vision_context": vision_context})
        print(f"\nProponowana akcja: {planned_action}")
        validation = guardian.validate(step, planned_action)
        if validation == "OK":
            print("Akcja zaakceptowana przez Guardian.")
            reflection_agent.add_mission_feedback(
                mission_description=step,
                feedback="Action succeeded.",
                success=True
            )
        else:
            print(f"Guardian odrzucił akcję: {validation}")
            reflection_agent.add_mission_feedback(
                mission_description=step,
                feedback=f"Action rejected: {validation}",
                success=False
            )
            break

if __name__ == "__main__":
    # Example usage:
    run_mission("Leć do przodu i wyląduj.", image_path="frame_2.png")
