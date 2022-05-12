import time
import getpass
import os


class Timer(object):
    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        print(f"[Timer]: {self.message}...")
        self.start_time = time.time()

    def __exit__(self, etype, value, traceback):
        elapsed_time = time.time() - self.start_time
        print(f"[Timer]: {self.message} took {elapsed_time}s")


def get_new_log_path(log_folder="logs"):
    current_user = getpass.getuser()
    i = 1
    for experiment in os.listdir(
        os.path.join(os.getcwd(), log_folder)
    ):
        if (current_user in experiment):
            experiment_index = int(experiment.split('-')[1].replace(".json", ""))
            i = max(
                experiment_index + 1,
                i
            )
    return os.path.join(os.getcwd(), log_folder, f"{current_user}-{i}")
