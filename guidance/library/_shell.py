import subprocess
import os
import subprocess
import asyncio


def shell(command, safe=True, _parser_context=None):
    """ Execute a shell command on the local machine (with user confirmation by default).

    Parameters
    ----------
    command : str
        The command to execute.
    safe : bool
        If True, the user will be asked to confirm the command before it is executed.
    """
    
    partial_output = _parser_context['partial_output']
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
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
        pass

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
        pass

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


getch = _Getch()


class Shell:
    """ A stateful shell object can be used to execute commands and get the output.
    """

    def __init__(self):
        import pty # TODO: Make this work on Windows
        self.shell_cmd = os.environ.get('SHELL', '/bin/sh')
        self.master_fd, self.slave_fd = pty.openpty()
        my_env = os.environ.copy()
        self.p = subprocess.Popen(
            self.shell_cmd,
            preexec_fn=os.setsid,
            stdin=self.slave_fd,
            stdout=self.slave_fd,
            stderr=self.slave_fd,
            text=True,
            env=my_env
        )
        self._env_inited = False
        self.loop = asyncio.get_event_loop()
        self.loop.add_reader(self.master_fd, self.read)

        # an event that the command has finished
        self.command_finished = asyncio.Event()
        self._current_output = ""

    def read(self):
        """ Read the next chunk of output from the shell.
        """
        new_text = os.read(self.master_fd, 10240).decode()
        self._current_output += new_text
        if self._current_output.endswith('autosh$ '):
            self.command_finished.set()
    
    async def __call__(self, command):
        ''' Send a command to the shell and return the output.
        '''

        # set up the environment if it hasn't been done yet
        if not self._env_inited:
            self._env_inited = True
            await self('export set PS1="autosh$ "')
        
        # send the command to the shell
        command = command + '\n'
        os.write(self.master_fd, command.encode())

        # wait for the command to finish
        await self.command_finished.wait()
        self.command_finished.clear()

        # return the collected output
        out = self._current_output
        self._current_output = ""
        return out