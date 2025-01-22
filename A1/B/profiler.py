import timeit
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import threading
import tabulate as tb


class ProProfiler:

    def __init__(self, interval=0.5): 
        self.start_time = None
        self.interval = interval
        self.samples = []
        self._thread = None
        self._thead_stop_event = threading.Event()
        
    def __enter__(self):
        self.samples = []
        self.start_time = timeit.default_timer()
        psutil.cpu_percent(percpu=True)
        self._thead_stop_event.clear()
        self._thread = threading.Thread(target=self.cpu_usage)
        self._thread.start()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self._thead_stop_event.set()
        self._thread.join()
        
    def cpu_usage(self):
        while not self._thead_stop_event.is_set():
            timestamp = timeit.default_timer() - self.start_time
            cpu_percentage = psutil.cpu_percent(interval=0, percpu=True)
            self.samples.append((timestamp, cpu_percentage))
            time.sleep(self.interval)

    def plot(self):
        timestamps = [sample[0] for sample in self.samples]
        cpu_data = [sample[1] for sample in self.samples]

        # Transpose to get per-core data
        core_data = np.array(cpu_data).T

        num_cores = core_data.shape[0]
        for i in range(num_cores):
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, core_data[i], label=f"Core {i}", marker='o')
            
            plt.title(f"CPU Usage Over Time - Core {i}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("CPU Usage (%)")
            plt.legend()
            plt.grid(True)

            plt.show()

        plt.figure(figsize=(10, 6))
        for i in range(num_cores):
            plt.plot(timestamps, core_data[i], label=f"Core {i}", marker='o')

        plt.title("CPU Usage Over Time - All Cores")
        plt.xlabel("Time (seconds)")
        plt.ylabel("CPU Usage (%)")
        plt.legend()
        plt.grid(True)

        plt.show()

    def summary_table(self):
        cpu_data = [sample[1] for sample in self.samples]

        core_data = np.array(cpu_data).T

        table_data = []
        for i, core in enumerate(core_data):
            min_val = np.min(core)
            avg_val = np.mean(core)
            std_val = np.std(core)
            max_val = np.max(core)
            time_above_50 = np.sum(core > 50) / len(core) * 100
            table_data.append([f"Core {i}", f"{avg_val:.2f}%", f"{max_val:.2f}%", f"{std_val:.2f}%", f"{time_above_50:.2f}%"])

        headers = ["Core", "Avg Usage", "Max Usage", "Deviation", "Time above 50%"]
        print(tb.tabulate(table_data, headers=headers, tablefmt="grid"))
