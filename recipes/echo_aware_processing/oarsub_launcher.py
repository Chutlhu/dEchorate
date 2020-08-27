import os
import signal
import socket
import subprocess
import pandas as pd

from tqdm import tqdm
from time import sleep
from itertools import product

SHORTNAME = 'dEchoRake'
CMD = 'python recipes/echo_aware_processing/main_dechorake_late.py '
ARR_IT = [0, 1, 2, 3, 4, 5]
DATA_IT = ['real', 'synt']
ROOM_IT = [1, 3, 5]
SNR_IT = [0, 10, 20]

N_SAMPLES = len(ARR_IT) * len(DATA_IT) * len(ROOM_IT) * len(SNR_IT)

###############
# AUXILIARY FUN
# usefull class that allows take actions after a timeout
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def execute(c):
    # return os.system(c)
    return subprocess.check_output(c, shell=True).rstrip()

# Asks a y/n question, with No as default.


def ask_binary_question(q):
    answer = input(q+" [y/N] ").lower()
    return (answer == 'yes' or answer == 'y')

# Generates the job "wrapper" command, i.e. including 'oarsub'


def gen_wrapper_command(cmd, shortname, nb_cores, max_duration_hours, path_to_folder):
    # Generate outputs
    outn = path_to_folder + shortname + "_%jobid%.out"
    errn = path_to_folder + shortname + "_%jobid%.err"
    # Generate wrappper command
    cmd = "oarsub -l "                       \
        + "/nodes=1"                      \
        + "/core=" + str(nb_cores) + "," \
        + "walltime=" + str(max_duration_hours) + ":00:00 "  \
        + "-S \"" + cmd + "\" " \
        + "-n " + shortname + " "       \
        + "-O " + outn + " "             \
        + "-E " + errn
    return cmd


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def oar_submit(wcmd):
    try:
        # let s wait 4 seconds before getting crazy...
        with timeout(seconds=4):
            # print("\t" + bcolors.BOLD + wcmd + bcolors.ENDC)
            execute(wcmd)
            # print("  submitted!")
            return 0
    except Exception as e:
        print(e)
        print(bcolors.WARNING
              + "MEEGA FAIL!!\n" + bcolors.ENDC
              + "\tNo worries, we will try it in a while..."
              )
        return oar_submit(wcmd)


def write_bash_file(command, filename):

    text = "# Enter the virtual env      \n"\
        + "source venv/bin/activate     \n"\
        + "								\n"\
        + "# Run the script\n"\
        + command

    # remove older version if exist
    execute("rm -f " + filename)
    with open(filename, "a") as f:
        f.write(text)
    # make the file executable
    execute("chmod +x " + filename)
    return filename


def main():
    print(bcolors.OKGREEN
          + "\n==================================================\n"
          + "=                    w|-|eLLc0me                 =\n"
          + "=               please help yourself             =\n"
          + "==================================================\n\n"
          + bcolors.ENDC)
    print("The magic trick of today is: \n\trunning "
          + bcolors.OKGREEN + CMD + bcolors.ENDC + "\n")

    print("So, you want to run some experiments, aren't you?")

    # print how many jobs it is going to be launched
    print(("You are going to submit %s %g jobs %s to IGRIDA:\n")
          % (bcolors.OKGREEN, N_SAMPLES, bcolors.ENDC))

    # ask for confirmation
    if not ask_binary_question("Wanna do it?"):
        print('Ahhh, it was nice. See ya later!')
        return
    print("\n")

    # job counter
    # submit jobs for all parameters combinations
    # load first pickle
    c = 0
    for arr in ARR_IT:
        for data in DATA_IT:
            for room in ROOM_IT:
                for snr in SNR_IT:

                        cmd = '%s --array %d --data %s --dataset %s --snr %d --rake 4' % (CMD, arr, data, room, snr)

                        # create bash wrapper script
                        filename = write_bash_file(cmd, "./run_job%d.sh" % (c))

                        #compute resources (n_core, time) for the job
                        n_cores, max_duration_hours = 2, 10
                        wcmd = gen_wrapper_command(
                            filename,                       # path to the binar
                            SHORTNAME + str(c),             # shortname for JOBID
                            n_cores, max_duration_hours,    # resources
                            './runs/')  # log directory

                        oar_submit(wcmd)
                        sleep(0.2)
                        c += 1

                        if c%25 == 0:
                            print('Submitted %d/%d' % (c, N_SAMPLES))

    print('Submitted %d/%d' % (c, N_SAMPLES))
    print(" SO LONG AND THANKS FOR THE FISH.")
# end main


if __name__ == "__main__":
    main()
