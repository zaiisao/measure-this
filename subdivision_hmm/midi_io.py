"""Minimal Standard MIDI File reader (note onsets and tempo map). Copied unchanged from
subdivision_refinement/scripts/midi_io.py so this experiment is self-contained."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import struct

@dataclass(frozen=True)
class MidiNote:
    tick: int
    time: float
    pitch: int
    velocity: int
    channel: int
    track: int

@dataclass
class MidiFile:
    path: str
    format: int
    tracks: int
    division: int
    notes: list[MidiNote]
    tempos: list[tuple[int,int]]
    duration: float

def _vlq(buf: bytes, i: int) -> tuple[int,int]:
    value=0
    while True:
        if i>=len(buf): raise ValueError('truncated variable-length quantity')
        b=buf[i]; i+=1; value=(value<<7)|(b&0x7f)
        if not b&0x80: return value,i

def read_midi(path: str|Path) -> MidiFile:
    path=Path(path); data=path.read_bytes()
    if data[:4]!=b'MThd' or len(data)<14: raise ValueError(f'not an SMF: {path}')
    hlen=struct.unpack('>I',data[4:8])[0]
    fmt,ntracks,division=struct.unpack('>HHH',data[8:14])
    if division&0x8000: raise ValueError('SMPTE timing is not supported')
    i=8+hlen; raw_notes=[]; raw_tempos=[]; last_tick=0
    for ti in range(ntracks):
        if data[i:i+4]!=b'MTrk': raise ValueError(f'missing MTrk at track {ti}: {path}')
        size=struct.unpack('>I',data[i+4:i+8])[0]; buf=data[i+8:i+8+size]; i+=8+size
        j=0; tick=0; running=None
        while j<len(buf):
            delta,j=_vlq(buf,j); tick+=delta; last_tick=max(last_tick,tick)
            if j>=len(buf): break
            first=buf[j]
            if first<0x80:
                if running is None: raise ValueError('running status without prior status')
                status=running
            else:
                status=first; j+=1
            if status==0xff:
                if j>=len(buf): break
                typ=buf[j]; j+=1; length,j=_vlq(buf,j); payload=buf[j:j+length]; j+=length
                if typ==0x51 and len(payload)==3:
                    raw_tempos.append((tick,int.from_bytes(payload,'big'),ti))
                if typ==0x2f: break
                continue
            if status in (0xf0,0xf7):
                length,j=_vlq(buf,j); j+=length; continue
            if 0xf8<=status<=0xfe:
                # MIDI realtime/system bytes do not consume channel data or alter running status.
                continue
            if status<0x80 or status>=0xf0: raise ValueError(f'unsupported status 0x{status:02x}')
            running=status; op=status&0xf0; ch=status&0x0f
            ndata=1 if op in (0xc0,0xd0) else 2
            if first<0x80: d1=first; j+=1
            else:
                if j>=len(buf): raise ValueError('truncated event')
                d1=buf[j]; j+=1
            d2=0
            if ndata==2:
                if j>=len(buf): raise ValueError('truncated event')
                d2=buf[j]; j+=1
            if op==0x90 and d2>0:
                raw_notes.append((tick,d1,d2,ch,ti))
            last_tick=max(last_tick,tick)
    # A tempo at tick zero is replaced by a file's own tempo at zero if present.
    tempo_map={0:500000}
    for tick,tempo,_ in sorted(raw_tempos,key=lambda x:(x[0],x[2])): tempo_map[tick]=tempo
    tempos=sorted(tempo_map.items())
    def to_seconds(tick:int)->float:
        total=0.0; prev=0; tempo=500000
        for t,tm in tempos:
            if t>=tick: break
            total+=(t-prev)*tempo/(division*1_000_000); prev=t; tempo=tm
        return total+(tick-prev)*tempo/(division*1_000_000)
    notes=[MidiNote(t,to_seconds(t),p,v,c,tr) for t,p,v,c,tr in raw_notes]
    notes.sort(key=lambda n:(n.time,n.track,n.channel,n.pitch))
    return MidiFile(str(path),fmt,ntracks,division,notes,tempos,to_seconds(last_tick))
