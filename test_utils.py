import time
import signal
from contextlib import contextmanager
from functools import wraps
from collections import defaultdict

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class Timed:
    """
    A versatile timer that can be used both as a function decorator and as a context manager.
    It measures the execution time of a function or a code block and prints the duration.

    As a decorator:
        @Timed()
        def some_function():
            # function implementation
        # Every time 'some_function' is called, it prints its execution duration.

    As a context manager:
        with Timed("custom_label"):
            # Code block whose duration you want to measure
        # Prints the execution duration of the code block with the specified label.

    Attributes:
        label (str, optional): The label to use for printing the execution time. 
                               Defaults to the function's name when used as a decorator.

    Methods:
        set_label(label): Sets a custom label for the timer.
    """
    _stats = defaultdict(lambda: {"calls": 0, "total_time": 0, "min_time": float('inf'), "max_time": 0})
    _console = Console()

    def __init__(self, label=None):
        self.label = label
        self.func_name = None  # Initialize func_name to store the function's name

    def __call__(self, func):
        self.func_name = func.__name__  # Store the name of the function

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

    @contextmanager
    def __enter__(self):
        self.start_time = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        duration = 1000 * (time.monotonic() - self.start_time)
        # Use func_name if available, else use the provided label
        label = self.func_name if self.func_name else self.label

        # Update stats
        stat = Timed._stats[label]
        stat["calls"] += 1
        stat["total_time"] += duration
        stat["min_time"] = min(stat["min_time"], duration)
        stat["max_time"] = max(stat["max_time"], duration)

        Timed._console.print(f"Duration of {label} = {duration:.2f}ms", style="bold green")
    
    @staticmethod
    def display_summary(sort_by="total_time"):
        if RICH_AVAILABLE:
            Timed._display_summary_rich(sort_by)
        else:
            Timed._display_summary_basic(sort_by)

    @staticmethod
    def _display_summary_rich(sort_by="total_time"):
        """
        Displays a summary table of execution times, sorted by the specified attribute.
        Available sorting options: 'total_time', 'calls', 'mean_time'.
        """
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Function", justify="left")
        table.add_column("Calls", justify="right")
        table.add_column("Total Time (ms)", justify="right")
        table.add_column("Min Time (ms)", justify="right")
        table.add_column("Max Time (ms)", justify="right")
        table.add_column("Mean Time (ms)", justify="right")

        for label, stats in sorted(Timed._stats.items(), key=lambda item: item[1][sort_by], reverse=True):
            mean_time = stats["total_time"] / stats["calls"]
            table.add_row(
                label,
                str(stats["calls"]),
                f"{stats['total_time']:.2f}",
                f"{stats['min_time']:.2f}",
                f"{stats['max_time']:.2f}",
                f"{mean_time:.2f}"
            )

        Timed._console.print(table)

    @staticmethod
    def _display_summary_basic(sort_by):
        print("Function\t\tCalls\tTotal Time (ms)\tMin Time (ms)\tMax Time (ms)\tMean Time (ms)")
        for label, stats in sorted(Timed._stats.items(), key=lambda item: item[1][sort_by], reverse=True):
            mean_time = stats["total_time"] / stats["calls"]
            print(f"{label}\t\t{stats['calls']}\t{stats['total_time']:.2f}\t"
                  f"{stats['min_time']:.2f}\t{stats['max_time']:.2f}\t{mean_time:.2f}")
    
    @staticmethod
    def setup_interrupt_handler():
        """
        Sets up a handler to ensure the summary is printed even if the script is terminated by CTRL-C.
        """
        def signal_handler(sig, frame):
            print("\nCTRL-C detected. Displaying summary before exit...")
            Timed.display_summary()
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)

    def set_label(self, label):
        """Sets a custom label for the timer."""
        self.label = label