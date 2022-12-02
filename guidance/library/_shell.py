import getpass
import sys
import subprocess
import shlex

def shell(command, partial_output, safe=True):
    """ Execute a shell command on the local machine (with user confirmation by default).
    """
    
    partial_output("{{execute '"+command+"'}}")
    
    # before running the command we need to get confirmation from user through keyboard input
    # this is to prevent accidental execution of commands
    if safe:
        c = ""
        while c != "\r" and c != "\n":
            c = getch()
            if c == '\x03':
                raise KeyboardInterrupt()
            elif c == "\x1b":
                partial_output("\nABORTED BY USER\n")
                return "{{execute '"+command+"'}}\n"+"ABORTED BY USER\n"

    # run the command
    all_output = "\n"
    partial_output(all_output)
    try:
        process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            output = process.stdout.readline().decode("utf-8") 
            all_output += output
            partial_output(output)
            if process.poll() is not None:
                break
        rc = process.poll()
        if rc != 0:
            raise Exception("Command failed")
    except Exception as e:
        all_output += str(e)+"\n"
        partial_output(str(e)+"\n")

    return "{{execute '"+command+"'}}"+all_output


# https://stackoverflow.com/questions/510357/how-to-read-a-single-character-from-the-user
class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


getch = _Getch()