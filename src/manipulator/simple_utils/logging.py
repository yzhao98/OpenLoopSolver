import io
import os
import select
import sys
import threading
import time
from typing import Union


class OutputGrabber:
    """
    This class can be used to capture the output of a stream (e.g. stdout, stderr).
    We redirect the stream to a pipe and read from the pipe in a separate thread.
    This makes it possible for further processing of the stream data.
    Currently, we read from the captured stream and write the data to the resource specified by a file descriptor fd.

    Usage:
    ------
    Examples:
    ---------
    with open("log.txt", "wb") as _f:
        with OutputGrabber(fd=_f.fileno(), stream=sys.stdout, duplicate=True):
            print("Hello World!")
            # Call shared library that prints to stdout.

    buffer = io.BytesIO()
    with OutputGrabber(fd=buffer, stream=sys.stdout, duplicate=True):
        ...
    buffer.seek(0)
    content = buffer.read()

    Limits:
    -------
    Cannot capture outputs when execute with pytest which has other capturing logics that conflict with this one.
     In tests, use fixture `capfd` provided by pytest instead.
    """

    def __init__(self, fd, stream=sys.stdout, duplicate=False, timeout=0.1):
        """
        Parameters
        ----------
        fd: a file descriptor or bytes buffer, the captured stream data will be written to this resource.
        stream: the stream to be captured.
        duplicate: if True, the captured stream data will also be written to the original stream.
        timeout: freq of reading from the pipe in seconds.
        """
        self.original_stream = stream
        self.original_stream_file_no = self.original_stream.fileno()

        # original_stream_dup_no will be a copy of original_stream_file_no:
        # the associated file descriptor point to the original IO source. (e.g. console)
        # stdout was replaced by pipe_in but console is still there.
        self.original_stream_dup_no = None
        # Create a pipe so the stream can be captured:
        self.pipe_out, self.pipe_in = os.pipe()
        self.fd: Union[int, io.BytesIO] = fd
        self.work_thread = None
        self.duplicate = duplicate
        self.finished = False
        self.timeout = timeout

    def _write(self, char: bytes):
        # support both file descriptor and bytes buffer:
        if isinstance(self.fd, int):
            os.write(self.fd, char)
        else:
            self.fd.write(char)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def start(self):
        """
        Start capturing the stream data.
        """
        # Save a copy of the stream:
        self.original_stream_dup_no = os.dup(self.original_stream_file_no)
        # original_stream is closed and replaced by pipe_in.
        os.dup2(self.pipe_in, self.original_stream_file_no)
        # We don't need the original file descriptor.
        os.close(self.pipe_in)
        # Now: (self.pipe_out, self.original_stream_file_no) are two ends of the pipe.
        # Start thread that will read the stream:
        self.work_thread = threading.Thread(target=self.read_output)
        self.work_thread.start()
        # Make sure that the thread is running and os.read() has executed:
        time.sleep(0.01)

    def stop(self):
        # original_stream: its file descriptor is replaced by pipe_in, so we are flushing new pipe here.
        self.original_stream.flush()
        # Note that (self.pipe_out, self.original_stream_file_no) are two ends of the pipe.
        # self.original_stream_file_no is the write-end. Closing it will cause os.read() to return.
        os.close(self.original_stream_file_no)
        self.finished = True
        self.work_thread.join()
        os.close(self.pipe_out)
        # Restore the original stream:
        os.dup2(self.original_stream_dup_no, self.original_stream_file_no)
        os.close(self.original_stream_dup_no)

    def read_output(self):
        poll = select.poll()
        poll.register(self.pipe_out, select.POLLIN)
        while True:
            # for a pipe, os.read() will block until there is data to read or the corresponding write-end is closed.
            # it will read at most 1024 bytes at a time.
            # For unknown reasons, the os.read does not return when the write-end of the pipe is closed.
            # Rescue:
            # 1. If we choose to close the read-end of the pipe (i.e. self.pipe_out),
            #   the os.read() will raise an exception. We may catch the exception and exit the loop.
            #   However, this might lead to losing outputs in the pipe.
            # 2. We can use select.select() to check if there is data to read to avoid blocking.
            #   If it is ready to read, we read it.
            #   Otherwise, we sleep for a while and check again (select sleep for us).
            #   If task is finished and the read-end is still not ready, we exit the loop.
            #   This approach adds a little overhead of waiting for at most timeout second. But it is acceptable.
            #   We adopt it here.
            #   Note that too small timeout will cause select.select being called too frequently.
            events = poll.poll(self.timeout * 1000)
            if len(events) == 0:
                if self.finished:
                    break
            for fd, event in events:
                if event & select.POLLIN:
                    char = os.read(self.pipe_out, 1024)
                    if not char:
                        break
                    self._write(char)
                    if self.duplicate:
                        os.write(self.original_stream_dup_no, char)
                if event & select.POLLHUP:  # when the write-end of the pipe is closed.
                    return
