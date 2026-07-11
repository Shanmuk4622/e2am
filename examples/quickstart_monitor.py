"""Monitor any code block — zero changes to the code itself.

Run:
    python examples/quickstart_monitor.py
"""

import time

from e2am import monitor


def heavy_work() -> None:
    """Stand-in for training, inference, data processing — anything."""
    total = 0
    deadline = time.time() + 5
    while time.time() < deadline:
        total += sum(i * i for i in range(10_000))


with monitor(project="quickstart", run_name="monitor-demo") as m:
    heavy_work()

result = m.result
print(f"\nTotal energy : {result.total_energy_wh:.4f} Wh")
print(f"Carbon       : {result.carbon.emissions_g:.4f} g CO2eq")
print(f"Artifacts    : {m.run_dir}")
