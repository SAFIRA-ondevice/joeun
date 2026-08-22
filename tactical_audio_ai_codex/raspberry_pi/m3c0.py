import struct
MAGIC=b"M3C0"; HEADER=struct.Struct("<4sHH")

def read_exact(serial,n):
    out=bytearray()
    while len(out)<n:
        chunk=serial.read(n-len(out))
        if not chunk: raise TimeoutError(f"serial timeout after {len(out)}/{n} bytes")
        out.extend(chunk)
    return bytes(out)

def sync_packet(serial,channels=3,max_samples=4096):
    window=bytearray(read_exact(serial,4))
    while bytes(window)!=MAGIC:
        window.pop(0); window.extend(read_exact(serial,1))
    seq,count=struct.unpack("<HH",read_exact(serial,4))
    if count<1 or count>max_samples: raise ValueError(f"invalid sample_count={count}")
    return seq,count,read_exact(serial,count*channels*2)
