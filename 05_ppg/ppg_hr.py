import time
import datetime
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks

from cms50d import CMS50D


def estimate_hr_with_fft(waveform, sampling_rate, min_bpm=40, max_bpm=220):
    waveform = np.asarray(waveform, dtype=float)

    if len(waveform) < max(32, int(sampling_rate * 4)):
        return None

    # Remove DC component
    waveform = waveform - np.mean(waveform)

    # Full FFT, then keep only positive frequencies
    fft_result = np.fft.fft(waveform)
    fft_freqs = np.fft.fftfreq(len(waveform), d=1.0 / sampling_rate)
    fft_magnitude = np.abs(fft_result)

    positive_mask = fft_freqs > 0
    fft_freqs = fft_freqs[positive_mask]
    fft_magnitude = fft_magnitude[positive_mask]

    # Keep only physiologically plausible HR range
    min_hz = min_bpm / 60.0
    max_hz = max_bpm / 60.0
    hr_mask = (fft_freqs >= min_hz) & (fft_freqs <= max_hz)

    if not np.any(hr_mask):
        return None

    valid_freqs = fft_freqs[hr_mask]
    valid_magnitude = fft_magnitude[hr_mask]

    if len(valid_magnitude) == 0:
        return None

    peak_index = np.argmax(valid_magnitude)
    peak_frequency = valid_freqs[peak_index]

    return peak_frequency * 60.0


def estimate_hr_with_peak_detection(waveform, sampling_rate, min_bpm=40, max_bpm=220):
    waveform = np.asarray(waveform, dtype=float)

    if len(waveform) < max(16, int(sampling_rate * 3)):
        return None

    # Remove DC component
    waveform = waveform - np.mean(waveform)

    # Minimum distance between peaks based on max HR
    min_distance = max(1, int(sampling_rate * 60.0 / max_bpm))

    # Adaptive prominence
    prominence = max(1.0, 0.2 * np.std(waveform))

    peaks, _ = find_peaks(
        waveform,
        distance=min_distance,
        prominence=prominence
    )

    if len(peaks) < 2:
        return None

    peak_intervals = np.diff(peaks) / sampling_rate
    avg_peak_interval = np.mean(peak_intervals)

    if avg_peak_interval <= 0:
        return None

    hr_bpm = 60.0 / avg_peak_interval

    if hr_bpm < min_bpm or hr_bpm > max_bpm:
        return None

    return hr_bpm


monitor = CMS50D(port="HID")
monitor.connect()
monitor.start_live_acquisition()

plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))

window_seconds = 10
target_fps = 30
frame_interval = 1.0 / target_fps

xdata = deque()
ydata = deque()

line, = ax.plot_date([], [], fmt='-m', label='Pulse Waveform')
text_hr = ax.text(0.02, 0.95, 'HR: -- bpm', transform=ax.transAxes, color='red', fontsize=12)
text_hr_fft = ax.text(0.02, 0.90, 'HR (FFT): -- bpm', transform=ax.transAxes, color='darkred', fontsize=12)
text_hr_peak = ax.text(0.02, 0.85, 'HR (Peak): -- bpm', transform=ax.transAxes, color='purple', fontsize=12)
text_spo2 = ax.text(0.02, 0.80, 'SpO2: --%', transform=ax.transAxes, color='blue', fontsize=12)

ax.set_ylim(0, 128)
ax.set_ylabel("Waveform")
ax.set_xlabel("Time")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.title("CMS50D Live Data")
plt.legend()

last_hr = None
last_spo2 = None
last_mode = "--"
last_draw_time = 0.0

sampling_rate = getattr(monitor, "sample_rate_hz", 60.0)

try:
    while True:
        got_data = False

        # Drain the queue quickly
        while True:
            data = monitor.get_latest_data()
            if data is None:
                break

            got_data = True
            now = data["timestamp"]
            waveform = data["waveform"]

            if waveform is not None:
                xdata.append(now)
                ydata.append(waveform)

            if data["pulse_rate"] is not None:
                last_hr = data["pulse_rate"]

            if data["spO2"] is not None:
                last_spo2 = data["spO2"]

            last_mode = data.get("mode", "--")

        # Keep only the last N seconds
        if xdata:
            cutoff = xdata[-1] - datetime.timedelta(seconds=window_seconds)
            while xdata and xdata[0] < cutoff:
                xdata.popleft()
                ydata.popleft()

        current_time = time.perf_counter()
        if current_time - last_draw_time >= frame_interval and xdata:
            line.set_data(list(xdata), list(ydata))
            ax.set_xlim(xdata[0], xdata[-1])

            hr_fft = estimate_hr_with_fft(list(ydata), sampling_rate)
            hr_peak = estimate_hr_with_peak_detection(list(ydata), sampling_rate)

            text_hr.set_text(f"HR: {last_hr if last_hr is not None else '--'} bpm")
            text_hr_fft.set_text(f"HR (FFT): {hr_fft:.1f} bpm" if hr_fft is not None else "HR (FFT): -- bpm")
            text_hr_peak.set_text(f"HR (Peak): {hr_peak:.1f} bpm" if hr_peak is not None else "HR (Peak): -- bpm")
            text_spo2.set_text(f"SpO2: {last_spo2 if last_spo2 is not None else '--'}%")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            last_draw_time = current_time

        if not got_data:
            plt.pause(0.005)
        else:
            plt.pause(0.001)

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    monitor.stop_live_acquisition()
    monitor.disconnect()
    plt.ioff()
    plt.show()