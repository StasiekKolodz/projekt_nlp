import requests

API_URL = "http://localhost:5002"

def takeoff(altitude=2.0):
    try:
        resp = requests.post(f"{API_URL}/takeoff", json={"altitude": altitude}, timeout=5)
        if resp.ok:
            return "Drone taking off."
        return f"Takeoff failed: {resp.text}"
    except Exception as e:
        return f"Takeoff error: {e}"

def fly_to(north: float, east: float, down: float):
    try:
        resp = requests.post(f"{API_URL}/goto_relative", json={"north": north, "east": east, "down": down}, timeout=5)
        if resp.ok:
            return f"Drone flying to (N:{north}, E:{east}, D:{down})"
        return f"Fly to failed: {resp.text}"
    except Exception as e:
        return f"Fly to error: {e}"

def land():
    try:
        resp = requests.post(f"{API_URL}/land", timeout=5)
        if resp.ok:
            return "Drone landing."
        return f"Land failed: {resp.text}"
    except Exception as e:
        return f"Land error: {e}"
