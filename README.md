# Projekt NLP: Agent System Documentation

## Overview

This project implements a multi-agent system for drone mission planning, vision analysis, navigation, and safety validation. Each agent runs in its own thread, communicates via a shared, thread-safe `MessagePool`, and is responsible for a specific aspect of the mission. The system is designed for modularity, extensibility, and robust inter-agent communication.

---

## Agent Architecture

### 1. MissionPlannerAgent (`agents/mission_planner.py`)

**Purpose:**  
Plans the drone mission by converting operator commands into a sequence of actionable steps.

**Key Functions:**

- `__init__(self, message_pool)`:  
  Initializes the agent with a language model, memory, and a reference to the shared `MessagePool`.

- `chat(self, user_input: str)`:  
  Handles interactive chat with the user, using a conversational agent with memory. Returns the agent's response.

- `request_mission(self, arg=None)`:  
  Posts a `"plan_mission"` message to the `MessagePool` with the operator's command. This triggers the mission planning process.

- `plan_mission(self, operator_command: str)`:  
  Uses an LLM to break down the operator's command into a list of mission steps (subgoals). Returns the plan as a list of steps.

- `read_messages(self)`:  
  Continuously polls the `MessagePool` for `"plan_mission"` messages. When found, generates a mission plan and posts a `"mission_steps"` message (with an empty vision context initially).

- `run(self)`:  
  Starts the message polling thread and enters a user input loop for interactive mission planning.

---

### 2. VisionAgent (`agents/vision_agent.py`)

**Purpose:**  
Analyzes images from the drone's camera and provides spatial context for navigation.

**Key Functions:**

- `__init__(self, message_pool=None)`:  
  Initializes the agent with a reference to the `MessagePool` and sets up the vision API endpoint.

- `describe_image(self, image_path: str)`:  
  Sends the image to an external API (`/camera_image`) and returns the vision context (spatial description).

- `read_messages(self)`:  
  Continuously polls the `MessagePool` for `"mission_steps"` messages that lack a `"vision_context"`. When found, calls `describe_image`, adds the vision context, and reposts the updated message.

- `start(self)`:  
  Starts the message polling thread for vision processing.

---

### 3. NavigatorAgent (`agents/navigator.py`)

**Purpose:**  
Executes the planned mission steps, using the vision context to inform navigation actions.

**Key Functions:**

- `__init__(self, message_pool=None)`:  
  Initializes the agent with navigation tools and a reference to the `MessagePool`.

- `takeoff(self, altitude=2.0)`:  
  Posts a `"drone_action"` message to the pool to command the drone to take off.

- `fly_to(self, north=0.0, east=0.0, down=0.0)`:  
  Posts a `"drone_action"` message to move the drone to a relative position.

- `land(self, arg=None)`:  
  Posts a `"drone_action"` message to land the drone.

- `read_messages(self)`:  
  Continuously polls the `MessagePool` for `"mission_steps"` messages with a vision context. For each step, invokes the navigation agent and posts the result.

- `run_task(self, task)`:  
  Allows direct invocation of navigation actions with a given task.

- `start(self)`:  
  Starts the message polling thread for navigation.

---

### 4. GuardianAgent (`agents/guardian.py`)

**Purpose:**  
Validates planned drone actions for safety and logical correctness before execution.

**Key Functions:**

- `__init__(self, message_pool=None)`:  
  Initializes the agent with a language model and a reference to the `MessagePool`.

- `validate(self, mission_step, planned_action, parameters=None)`:  
  Uses an LLM to check if the planned action is logical and safe for the given mission step.

- `execute_action(self, action, parameters=None)`:  
  Calls the appropriate function from `drone_tools` to execute the validated action.

- `read_messages(self)`:  
  Continuously polls the `MessagePool` for `"drone_action"` messages. Validates each action, executes it if valid, and posts a `"guardian_validation"` message with the result.

- `start(self)`:  
  Starts the message polling thread for validation.

---

## MessagePool (`agents/message_pool.py`)

**Purpose:**  
A thread-safe, shared message bus for agent communication.

**Key Functions:**

- `build_message(self, msg_type, content)`:  
  Constructs a message dictionary.

- `post(self, message)`:  
  Adds a message to the pool.

- `get_all(self)`:  
  Returns a copy of all messages in the pool.

- `find(self, predicate)`:  
  Returns all messages matching a given predicate.

- `remove_type(self, msg_type)`:  
  Removes all messages of a given type.

- `remove_message(self, message)`:  
  Removes a specific message from the pool.

---

## Communication Flow

1. **Mission Planning:**  
   - The user interacts with `MissionPlannerAgent`, which posts a `"plan_mission"` message.
   - `MissionPlannerAgent` reads this message, generates a mission plan, and posts a `"mission_steps"` message with an empty vision context.

2. **Vision Context:**  
   - `VisionAgent` detects `"mission_steps"` messages without a vision context.
   - It calls the vision API, adds the vision context, and reposts the updated `"mission_steps"` message.

3. **Navigation:**  
   - `NavigatorAgent` waits for `"mission_steps"` messages with a vision context.
   - For each step, it invokes the navigation logic and posts a `"drone_action"` message.

4. **Validation and Execution:**  
   - `GuardianAgent` reads `"drone_action"` messages, validates them, and if valid, executes the action using `drone_tools`.
   - It posts a `"guardian_validation"` message with the result.

5. **Feedback Loop:**  
   - Each agent removes processed messages from the pool to prevent duplicate processing.
   - The system is fully asynchronous and thread-safe, allowing agents to work concurrently.

---

## Implementation Notes

- **Threading:**  
  Each agent runs its own polling thread, allowing for concurrent, event-driven processing.

- **Extensibility:**  
  New agents or message types can be added by following the same message-driven pattern.

- **Error Handling:**  
  Agents print errors and validation failures to the console for transparency.

- **API Integration:**  
  The vision agent uses an external API for image analysis, and the guardian agent uses `drone_tools` for real drone control.

---

## Example Workflow

1. User: `"Leć do przodu i wyląduj."`
2. `MissionPlannerAgent` → posts `"plan_mission"` → generates steps → posts `"mission_steps"` (no vision context)
3. `VisionAgent` → adds vision context → reposts `"mission_steps"`
4. `NavigatorAgent` → executes each step with vision context → posts `"drone_action"`
5. `GuardianAgent` → validates and executes action → posts `"guardian_validation"`

---

## Summary

This system demonstrates a robust, modular, and extensible architecture for multi-agent collaboration in a drone mission scenario. Each agent is responsible for a single aspect, and all communication is handled via a thread-safe message pool, ensuring decoupling and scalability.