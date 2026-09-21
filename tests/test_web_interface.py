import urllib.request
import json
import time
import sys

BASE_URL = "http://127.0.0.1:8080"

def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req, timeout=3) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        return resp.read().decode("utf-8"), dict(resp.headers)

def post_json(path, data):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=3) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        res = json.loads(resp.read().decode("utf-8"))
    time.sleep(0.02)
    return res

def main():
    print("[1] Testing GET / (Minimal HTML Interface)...")
    html, _ = get("/")
    assert "<title>Radar Shader Controls</title>" in html
    assert "scanning_speed" in html
    assert "decay_time" in html
    assert "scale" in html
    assert "auto_speed" in html
    assert "auto_freq" in html
    assert "multiplySpeed(2.0)" in html
    assert "multiplySpeed(0.25)" in html
    # Verify new features in HTML
    assert "setTopology" in html, "HTML should contain setTopology"
    assert "topoBtnPolar" in html, "HTML should contain polar topology button"
    assert "mutateCandidate" in html, "HTML should contain mutateCandidate button"
    assert "cancelCandidate" in html, "HTML should contain cancelCandidate button"
    assert "reinforceCandidate" in html, "HTML should contain reinforceCandidate button"
    assert "triggerImprint" in html, "HTML should contain Easter egg triggerImprint"
    assert "imprintInput" in html, "HTML should contain arbitrary message input"
    assert "<canvas" not in html, "Canvas element should remain removed for minimal UI"
    print("  -> Passed. Minimal HTML contains Topology, Genetic Evolution, and Easter Egg Imprint.")

    print("[2] Testing GET /api/status...")
    status_str, _ = get("/api/status")
    status = json.loads(status_str)
    print(f"  -> Initial status: {status}")
    assert "scale" in status
    assert "scale_type" in status
    assert "auto_speed" in status
    assert "rhythm_division" in status
    assert "auto_freq" in status
    assert "nca_topology" in status
    assert "topology" in status
    assert "genetic_generation" in status
    assert "genetic_has_candidate" in status
    assert "genetic_mutated_count" in status
    assert "genetic_mutation_scale" in status
    assert "genetic_mutation_rate" in status
    print("  -> Passed. Status API returns all autotune, topology, and genetic fields.")

    print("[3] Testing Frequency Scales (log, mel, linear)...")
    res = post_json("/api/controls", {"scale": "mel"})
    assert res["scale"] == "mel" and res["scale_type"] == 1, f"Expected mel/1, got {res}"
    print("  -> Mel scale verified.")

    res = post_json("/api/controls", {"scale": "linear"})
    assert res["scale"] == "linear" and res["scale_type"] == 2, f"Expected linear/2, got {res}"
    print("  -> Linear scale verified.")

    res = post_json("/api/controls", {"scale": "log"})
    assert res["scale"] == "log" and res["scale_type"] == 0, f"Expected log/0, got {res}"
    print("  -> Log scale verified.")

    print("[4] Testing Rotation Speed Multipliers (2x, 4x faster / slower)...")
    post_json("/api/controls", {"scanning_speed": 0.40, "auto_speed": False})
    
    res = post_json("/api/controls", {"speed_multiplier": 2.0})
    assert abs(res["scanning_speed"] - 0.80) < 0.01, f"Expected 0.80, got {res['scanning_speed']}"
    print(f"  -> 2x faster multiplier: speed = {res['scanning_speed']} rev/s")

    res = post_json("/api/controls", {"speed_multiplier": 4.0})
    assert abs(res["scanning_speed"] - 3.20) < 0.02, f"Expected 3.20, got {res['scanning_speed']}"
    print(f"  -> 4x faster multiplier: speed = {res['scanning_speed']} rev/s")

    res = post_json("/api/controls", {"speed_multiplier": 0.5})
    assert abs(res["scanning_speed"] - 1.60) < 0.02, f"Expected 1.60, got {res['scanning_speed']}"
    print(f"  -> 1/2x slower multiplier: speed = {res['scanning_speed']} rev/s")

    res = post_json("/api/controls", {"speed_multiplier": 0.25})
    assert abs(res["scanning_speed"] - 0.40) < 0.02, f"Expected 0.40, got {res['scanning_speed']}"
    print(f"  -> 1/4x slower multiplier: speed = {res['scanning_speed']} rev/s")

    print("[5] Testing Autotune options...")
    res = post_json("/api/controls", {"auto_speed": True, "rhythm_division": 2, "auto_freq": True})
    assert res["auto_speed"] is True
    assert res["auto_freq"] is True
    assert res["rhythm_division"] == 2
    print(f"  -> Auto speed & auto freq enabled. Detected BPM: {res.get('detected_bpm')}")

    print("[6] Testing NCA Topology Controls (Cartesian <-> Polar with pole in center)...")
    # Set to Polar via /api/topology
    res = post_json("/api/topology", {"topology": "polar"})
    assert abs(res["nca_topology"] - 1.0) < 0.01, f"Expected 1.0, got {res['nca_topology']}"
    assert res["topology"] == "polar", f"Expected polar, got {res['topology']}"
    print(f"  -> Topology Polar verified: nca_topology={res['nca_topology']}")

    # Set to Cartesian via /api/topology
    res = post_json("/api/topology", {"topology": "cartesian"})
    assert abs(res["nca_topology"] - 0.0) < 0.01, f"Expected 0.0, got {res['nca_topology']}"
    assert res["topology"] == "cartesian", f"Expected cartesian, got {res['topology']}"
    print(f"  -> Topology Cartesian verified: nca_topology={res['nca_topology']}")

    # Set float topology via /api/controls
    res = post_json("/api/controls", {"nca_topology": 0.5})
    assert abs(res["nca_topology"] - 0.5) < 0.01, f"Expected 0.5, got {res['nca_topology']}"
    print(f"  -> Topology float value verified: nca_topology={res['nca_topology']}")

    print("[7] Testing Easter Egg Structure Imprint (/api/imprint with arbitrary message)...")
    res = post_json("/api/imprint", {"message": "BYOB"})
    assert res["status"] == "imprinted", f"Expected imprinted, got {res}"
    assert res["message"] == "BYOB"
    print("  -> Default Easter Egg 'BYOB' imprint request verified.")

    res = post_json("/api/imprint", {"message": "ANTIGRAVITY"})
    assert res["status"] == "imprinted"
    assert res["message"] == "ANTIGRAVITY"
    print("  -> Arbitrary message 'ANTIGRAVITY' imprint request verified.")

    print("[8] Testing Human-Assisted Genetic Evolution (/api/genetic)...")
    # 8.1 Reset to Generation 0
    res = post_json("/api/genetic", {"action": "reset"})
    assert res["genetic_generation"] == 0, f"Expected Gen 0, got {res['genetic_generation']}"
    assert res["genetic_has_candidate"] is False
    print("  -> Gen 0 reset verified.")

    # 8.2 Mutate candidate
    res = post_json("/api/genetic", {
        "action": "mutate",
        "target": "all",
        "scale": 0.15,
        "rate": 0.25
    })
    # Wait a moment for engine tick if needed
    time.sleep(0.05)
    status_str, _ = get("/api/status")
    status = json.loads(status_str)
    assert status["genetic_has_candidate"] is True, f"Expected candidate live, got {status}"
    assert status["genetic_mutated_count"] > 0, f"Expected mutated weights > 0, got {status['genetic_mutated_count']}"
    print(f"  -> Mutation produced candidate with {status['genetic_mutated_count']} perturbed weights.")

    # 8.3 Cancel candidate (reverts to baseline)
    res = post_json("/api/genetic", {"action": "cancel"})
    time.sleep(0.05)
    status_str, _ = get("/api/status")
    status = json.loads(status_str)
    assert status["genetic_has_candidate"] is False, f"Expected candidate cancelled, got {status}"
    assert status["genetic_generation"] == 0
    print("  -> Cancelled candidate; reverted to Gen 0 baseline.")

    # 8.4 Mutate perception subsystem and Reinforce to Gen 1
    post_json("/api/genetic", {
        "action": "mutate",
        "target": "perception",
        "scale": 0.20,
        "rate": 0.30
    })
    time.sleep(0.05)
    res = post_json("/api/genetic", {"action": "reinforce"})
    time.sleep(0.05)
    status_str, _ = get("/api/status")
    status = json.loads(status_str)
    assert status["genetic_generation"] == 1, f"Expected Gen 1, got {status['genetic_generation']}"
    assert status["genetic_has_candidate"] is False
    print(f"  -> Reinforced candidate! Advanced to Gen {status['genetic_generation']}.")

    # 8.5 Rollback to Gen 0
    res = post_json("/api/genetic", {"action": "rollback"})
    time.sleep(0.05)
    status_str, _ = get("/api/status")
    status = json.loads(status_str)
    assert status["genetic_generation"] == 0, f"Expected rollback to Gen 0, got {status['genetic_generation']}"
    print("  -> Rollback verified; restored previous generation.")

    print("\nALL TOPOLOGY, GENETIC ALGORITHM & EASTER EGG IMPRINT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
