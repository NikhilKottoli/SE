import time

def run_worker():
    while True:
        print("Worker: Running scheduled tasks...")
        time.sleep(10)

if __name__ == "__main__":
    run_worker()
