import asyncio
import httpx
import time
import json

URL = "http://localhost:5000/verify"
CONCURRENT_REQUESTS = 100

TEST_PAYLOAD = {
    "transcript": "Agent: Hi, am I speaking to PAWAN KUMAR? User: Yes, this is Pawan."
}

async def send_request(client, request_id):
    start_time = time.time()
    try:
        print(f"[{request_id}] Sending request...")
        response = await client.post(URL, json=TEST_PAYLOAD, timeout=120)
        latency = time.time() - start_time
        if response.status_code == 200:
            print(f"[{request_id}] Success! Latency: {latency:.2f}s")
            return latency
        else:
            print(f"[{request_id}] Failed! Status: {response.status_code}")
            return None
    except Exception as e:
        print(f"[{request_id}] Error: {e}")
        return None

async def run_stress_test():
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, i) for i in range(CONCURRENT_REQUESTS)]
        print(f"--- Starting Stress Test with {CONCURRENT_REQUESTS} Concurrent Requests ---")
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        success_latencies = [l for l in results if l is not None]
        print("\n--- Stress Test Results ---")
        print(f"Total Requests: {CONCURRENT_REQUESTS}")
        print(f"Successes: {len(success_latencies)}")
        print(f"Failures: {CONCURRENT_REQUESTS - len(success_latencies)}")
        print(f"Total Test Duration: {total_time:.2f}s")
        if success_latencies:
            print(f"Average Latency: {sum(success_latencies) / len(success_latencies):.2f}s")
            print(f"Min Latency: {min(success_latencies):.2f}s")
            print(f"Max Latency: {max(success_latencies):.2f}s")

if __name__ == "__main__":
    asyncio.run(run_stress_test())
