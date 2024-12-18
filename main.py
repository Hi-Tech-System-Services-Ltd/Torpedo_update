import subprocess
import sys
import multiprocessing
from pathlib import Path
import time

def run_script(script_path):
    while True:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(f"[{Path(script_path).name}]: {output.strip()}")

                error = process.stderr.readline()
                if error:
                    print(f"[{Path(script_path).name} ERROR]: {error.strip()}", file=sys.stderr)

            process.wait()

            if process.returncode != 0:
                print(f"[{Path(script_path).name}] exited with error code {process.returncode}", file=sys.stderr)

        except KeyboardInterrupt:
            print(f"KeyboardInterrupt received. Terminating {Path(script_path).name}.")
            process.terminate()
            break

        except Exception as e:
            print(f"An unexpected error occurred in {Path(script_path).name}: {e}", file=sys.stderr)
            process.terminate()

        finally:
            process.terminate()
            process.wait()

        print(f"Restarting {Path(script_path).name} in 1 second...")
        time.sleep(1)

if __name__ == "__main__":
    current_dir = Path(__file__).parent.absolute()
    scripts = [
        current_dir / "detection_app.py",
        current_dir / "detection_app.py",
        current_dir / "detection_app.py"
    ]

    processes = []
    for script in scripts:
        p = multiprocessing.Process(target=run_script, args=(str(script),))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Terminating all scripts...")
        for p in processes:
            p.terminate()
            p.join()
