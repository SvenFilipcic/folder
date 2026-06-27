"""Raw-socket IPC: length-prefixed pickle messages over localhost TCP.

TCP is a byte stream, not a message stream, so every message is framed as
[4-byte big-endian length][pickle payload]. Both the inference server
(workers/infer_server.py, env `infer`) and the sim client (head_parallel.py,
env `fold`) import this module.
"""

import pickle
import socket
import struct

_LEN = struct.Struct(">I")  # 4-byte big-endian unsigned length prefix


def _recv_exact(sock, n):
    """Read exactly n bytes or raise ConnectionError if the peer closes."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed during recv")
        got += k
    return bytes(buf)


def send_msg(sock, obj):
    """Pickle obj and send it framed with a 4-byte length prefix."""
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_LEN.pack(len(payload)) + payload)


def recv_msg(sock):
    """Receive one framed message and unpickle it."""
    (n,) = _LEN.unpack(_recv_exact(sock, _LEN.size))
    return pickle.loads(_recv_exact(sock, n))


def connect(host="127.0.0.1", port=5557, retries=60, delay=1.0):
    """Client side: connect to the server, retrying while it boots.

    The server loads two checkpoints + Torch before it listens, which can take
    tens of seconds, so the client retries rather than failing on the first
    refused connection.
    """
    import time
    last = None
    for _ in range(retries):
        try:
            s = socket.create_connection((host, port))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return s
        except OSError as e:
            last = e
            time.sleep(delay)
    raise ConnectionError(f"could not connect to {host}:{port}: {last}")


def serve(port, handler, host="127.0.0.1"):
    """Server side: accept one client at a time and dispatch each message.

    `handler(msg) -> reply` is called per message. If the handler returns the
    sentinel string "__SHUTDOWN__" the server stops after replying.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[socket_ipc] listening on {host}:{port}", flush=True)
    try:
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[socket_ipc] client connected: {addr}", flush=True)
            try:
                while True:
                    try:
                        msg = recv_msg(conn)
                    except (ConnectionError, EOFError):
                        print("[socket_ipc] client disconnected", flush=True)
                        break
                    reply = handler(msg)
                    if reply == "__SHUTDOWN__":
                        send_msg(conn, {"ok": True})
                        return
                    send_msg(conn, reply)
            finally:
                conn.close()
    finally:
        srv.close()
