import json
import httpx


def main():
    url = "http://localhost:8000/api/chat/stream"
    payload = {
        "message": "电压是什么？",
        "thread_id": "docker-test-001",
    }

    with httpx.Client(timeout=None) as client:
        with client.stream("POST", url, json=payload) as resp:
            print(f"HTTP {resp.status_code}")
            event_type = None
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    print(f"[{event_type}] {data}")


if __name__ == "__main__":
    main()