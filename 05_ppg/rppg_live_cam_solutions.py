import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import time


# =========================
# Matplotlib live plot
# =========================
plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))

xdata = []
ydata = []
ydata_filtered = []

line_raw, = ax.plot([], [], label="Raw green signal", alpha=0.5)
line_filt, = ax.plot([], [], label="Filtered signal", linewidth=2)

text_hr = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="red", fontsize=12, va="top")
text_status = ax.text(0.02, 0.78, "", transform=ax.transAxes, color="blue", fontsize=11, va="top")

ax.set_ylabel("Signal")
ax.set_xlabel("Time (s)")
plt.title("rPPG")
plt.legend()


# =========================
# Face detector
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# =========================
# Signal processing
# =========================
def normalize_signal(signal):
    signal = np.asarray(signal, dtype=float)
    mean = np.mean(signal)
    std = np.std(signal)

    if std < 1e-8:
        return signal - mean

    return (signal - mean) / std

def bandpass_filter(signal, fs=30, min_bpm=40, max_bpm=220):

    lowcut = min_bpm / 60.0
    highcut = max_bpm / 60.0
    try:
        b, a = butter(4, [lowcut, highcut], btype="band", analog=False, fs=fs)
        signal = np.asarray(signal, dtype=float)

        padlen = 3 * max(len(a), len(b))
        if len(signal) <= padlen:
            return signal

        return filtfilt(b, a, signal)
    except Exception as e:
        print(f"[ERROR] Filter failed: {e}")
        return np.asarray(signal, dtype=float)

def estimate_hr_with_fft(waveform, sampling_rate, min_bpm=40, max_bpm=220):
    waveform = np.asarray(waveform, dtype=float)

    if len(waveform) < int(sampling_rate * 4):
        return None

    # Remove DC component
    waveform = waveform - np.mean(waveform)

    # Real FFT
    fft_result = np.fft.rfft(waveform)
    fft_freqs = np.fft.rfftfreq(len(waveform), d=1.0 / sampling_rate)
    fft_magnitude = np.abs(fft_result)

    # Keep only physiologically plausible HR range
    min_hz = min_bpm / 60.0
    max_hz = max_bpm / 60.0
    mask = (fft_freqs >= min_hz) & (fft_freqs <= max_hz)

    if not np.any(mask):
        return None

    valid_freqs = fft_freqs[mask]
    valid_mag = fft_magnitude[mask]

    if len(valid_mag) == 0:
        return None

    peak_index = np.argmax(valid_mag)
    peak_frequency = valid_freqs[peak_index]

    return peak_frequency * 60.0


def estimate_hr_with_peak_detection(waveform, sampling_rate, min_bpm=40, max_bpm=220):
    waveform = np.asarray(waveform, dtype=float)

    if len(waveform) < int(sampling_rate * 4):
        return None

    #remove DC
    waveform = waveform - np.mean(waveform)

    min_distance = max(1, int(sampling_rate * 60.0 / max_bpm))
    prominence = max(0.1, 0.2 * np.std(waveform))

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


# =========================
# Acquisition parameters
# =========================
last_estimation_time = 0.0
estimation_interval = 1.0  # seconds

hr_fft = None
hr_peak = None

# Use a slightly higher target FPS if the camera can sustain it
cap = cv2.VideoCapture(0)
target_fps = 20.0
frame_interval = 1.0 / target_fps

# ROI smoothing
prev_roi = None
alpha_roi = 0.8  # higher = more stable ROI

# Plot and HR windows
plot_window_seconds = 20.0
estimation_window_seconds = 10.0


# =========================
# Main loop
# =========================
while True:
    loop_start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(300, 300))

    face_found = False

    for (x, y, w, h) in faces:
        face_found = True

        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Forehead ROI: narrower and more central for stability
        x_roi = x + int(0.30 * w)
        w_roi = int(0.40 * w)
        y_roi = y + int(0.08 * h)
        h_roi = int(0.18 * h)

        # Smooth ROI over time
        if prev_roi is None:
            prev_roi = (x_roi, y_roi, w_roi, h_roi)
        else:
            px, py, pw, ph = prev_roi
            x_roi = int(alpha_roi * px + (1.0 - alpha_roi) * x_roi)
            y_roi = int(alpha_roi * py + (1.0 - alpha_roi) * y_roi)
            w_roi = int(alpha_roi * pw + (1.0 - alpha_roi) * w_roi)
            h_roi = int(alpha_roi * ph + (1.0 - alpha_roi) * h_roi)
            prev_roi = (x_roi, y_roi, w_roi, h_roi)

        # Keep ROI inside frame bounds
        x_roi = max(0, x_roi)
        y_roi = max(0, y_roi)
        w_roi = max(1, min(w_roi, frame.shape[1] - x_roi))
        h_roi = max(1, min(h_roi, frame.shape[0] - y_roi))

        roi = frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi, :]
        cv2.rectangle(frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 0, 255), 2)

        if roi.size == 0:
            break

        # Mean green intensity
        green_channel = roi[:, :, 1]
        signal_value = float(np.mean(green_channel))

        xdata.append(loop_start)
        ydata.append(signal_value)

        # Keep only the last plot window
        cutoff = loop_start - plot_window_seconds
        xdata = [t for t in xdata if t >= cutoff]
        ydata = ydata[-len(xdata):]

        # Estimate HR once per interval
        if len(ydata) > 30 and (loop_start - last_estimation_time) >= estimation_interval:
            # Use only the last estimation window for HR estimation
            estimation_cutoff = loop_start - estimation_window_seconds
            estimation_x = [t for t in xdata if t >= estimation_cutoff]
            estimation_y = ydata[-len(estimation_x):]

            if len(estimation_y) > 30:
                dt = estimation_x[-1] - estimation_x[0]

                if dt > 0:
                    sampling_rate = len(estimation_x) / dt

                    # Normalize first, then bandpass filter
                    normalized_signal = normalize_signal(estimation_y)
                    filtered_signal = bandpass_filter(
                        normalized_signal,
                        fs=sampling_rate,
                        min_bpm=40, max_bpm=220)

                    hr_fft = estimate_hr_with_fft(filtered_signal, sampling_rate)
                    hr_peak = estimate_hr_with_peak_detection(filtered_signal, sampling_rate)

                    ydata_filtered = filtered_signal

                    # Update Matplotlib plot
                    plot_x = np.array(estimation_x) - estimation_x[0]
                    line_raw.set_data(plot_x, normalized_signal)
                    line_filt.set_data(plot_x, ydata_filtered)

                    ax.set_xlim(plot_x[0], plot_x[-1])

                    combined = np.concatenate([normalized_signal, ydata_filtered])
                    y_min = float(np.min(combined))
                    y_max = float(np.max(combined))

                    if y_min == y_max:
                        ax.set_ylim(y_min - 1.0, y_max + 1.0)
                    else:
                        margin = 0.15 * (y_max - y_min)
                        ax.set_ylim(y_min - margin, y_max + margin)

                    fft_text = f"{hr_fft:.1f}" if hr_fft is not None else "--"
                    peak_text = f"{hr_peak:.1f}" if hr_peak is not None else "--"

                    text_hr.set_text(
                        f"HR (FFT): {fft_text} bpm\n"
                        f"HR (Peak): {peak_text} bpm\n"
                    )

                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    last_estimation_time = loop_start

        # Frame overlay
        fft_overlay = f"{hr_fft:.1f}" if hr_fft is not None else "--"
        peak_overlay = f"{hr_peak:.1f}" if hr_peak is not None else "--"

        cv2.putText(
            frame,
            f"HR (FFT): {fft_overlay} bpm",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"HR (Peak): {peak_overlay} bpm",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # Use only the first detected face
        break

    if not face_found:
        prev_roi = None
        cv2.putText(
            frame,
            "No face detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("rPPG", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    elapsed = time.time() - loop_start
    time.sleep(max(0, frame_interval - elapsed))

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()