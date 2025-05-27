import requests

API_URL = "http://localhost:5000"

def test_takeoff():
    resp = requests.post(f"{API_URL}/takeoff", json={"altitude": 2.5})
    print("Takeoff:", resp.status_code, resp.json())

def test_goto_relative():
    resp = requests.post(f"{API_URL}/goto_relative", json={"north": 5, "east": 0, "down": 0})
    print("Goto Relative:", resp.status_code, resp.json())

def test_land():
    resp = requests.post(f"{API_URL}/land")
    print("Land:", resp.status_code, resp.json())

def test_telemetry():
    resp = requests.get(f"{API_URL}/telemetry")
    print("Telemetry:", resp.status_code, resp.json())

if __name__ == "__main__":
    test_takeoff()
    test_goto_relative()
    test_telemetry()
    test_land()
