"""Feed a continuous recording to the models the way a live system would: half-second chunks, a rolling window."""
from enginetwin.latency import Stream


def rolling_windows(stream: Stream, chunk_s=0.5, window_s=2.0):
    """Yield (end_s, window, sound_chunk, vibration_chunk) every `chunk_s`, starting once `window_s` of signal has
    arrived. The window holds the latest `window_s` seconds; the chunks are only the newest `chunk_s`."""
    duration = len(stream.sound) / stream.fs
    steps = int(round((duration - window_s) / chunk_s)) + 1
    for k in range(steps):
        end = window_s + k * chunk_s
        stop = int(round(end * stream.fs))
        start = stop - int(round(chunk_s * stream.fs))
        yield end, stream.window(end, window_s), stream.sound[start:stop], stream.vibration[start:stop]
