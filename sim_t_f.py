"""Standing wave simulation for string tension vs frequency.

The model follows f = (n / (2 * l)) * sqrt(T / rho), where:
- n   : mode number (integer, n >= 1)
- l   : string length (m)
- T   : tension (N)
- rho : linear density (kg/m)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation


def frequency_from_tension(tension: np.ndarray, mode: int, length_m: float, linear_density: float) -> np.ndarray:
	"""Compute fundamental or harmonic frequency for a given tension array."""
	tension = np.asarray(tension, dtype=float)
	if np.any(tension <= 0):
		raise ValueError("Tension must be positive.")
	if mode < 1:
		raise ValueError("Mode number n must be >= 1.")
	if length_m <= 0:
		raise ValueError("String length must be positive.")
	if linear_density <= 0:
		raise ValueError("Linear density must be positive.")

	return (mode / (2.0 * length_m)) * np.sqrt(tension / linear_density)


def wavelength(mode: int, length_m: float) -> float:
	"""Compute wavelength for a given mode using lambda = 2l / n."""
	if mode < 1:
		raise ValueError("Mode number n must be >= 1.")
	if length_m <= 0:
		raise ValueError("String length must be positive.")
	return 2.0 * length_m / mode


def demo_plot(
	t_min: float = 1.0,
	t_max: float = 50.0,
	num_points: int = 200,
	mode: int = 1,
	length_m: float = 1.0,
	linear_density: float = 0.01,
) -> None:
	"""Plot frequency vs tension for chosen parameters."""

	tensions = np.linspace(t_min, t_max, num_points)
	freqs = frequency_from_tension(tensions, mode=mode, length_m=length_m, linear_density=linear_density)

	plt.figure(figsize=(7, 4))
	plt.plot(tensions, freqs, color="tab:blue")
	plt.title("Standing Wave on a String: f vs T")
	plt.xlabel("Tension T (N)")
	plt.ylabel("Frequency f (Hz)")
	plt.grid(True, ls="--", alpha=0.4)
	plt.tight_layout()
	plt.show()


def interactive_plot(
	t_min: float = 1.0,
	t_max: float = 50.0,
	num_points: int = 300,
	mode: int = 1,
	length_m: float = 1.0,
	linear_density: float = 0.01,
) -> None:
	"""Interactive sliders for mode, length, density, and tension range."""

	# Base data
	tensions = np.linspace(t_min, t_max, num_points)
	freqs = frequency_from_tension(tensions, mode=mode, length_m=length_m, linear_density=linear_density)

	fig, ax = plt.subplots(figsize=(8, 5))
	plt.subplots_adjust(left=0.1, bottom=0.32)
	[line] = ax.plot(tensions, freqs, color="tab:blue", lw=2)
	ax.grid(True, ls="--", alpha=0.4)
	ax.set_xlabel("Tension T (N)")
	ax.set_ylabel("Frequency f (Hz)")
	fig.suptitle("Standing Wave on a String: interactive f vs T")

	# Slider axes
	ax_color = "lightgoldenrodyellow"
	# Two-column layout for clearer grouping
	ax_mode = plt.axes([0.10, 0.23, 0.38, 0.03], facecolor=ax_color)
	ax_length = plt.axes([0.52, 0.23, 0.38, 0.03], facecolor=ax_color)
	ax_rho = plt.axes([0.10, 0.17, 0.38, 0.03], facecolor=ax_color)
	ax_tmin = plt.axes([0.52, 0.17, 0.18, 0.03], facecolor=ax_color)
	ax_tmax = plt.axes([0.72, 0.17, 0.18, 0.03], facecolor=ax_color)

	s_mode = Slider(ax_mode, "n (mode)", 1, 10, valinit=mode, valstep=1)
	s_length = Slider(ax_length, "length l (m)", 0.2, 5.0, valinit=length_m, valstep=0.01)
	s_rho = Slider(ax_rho, "rho (kg/m)", 0.001, 0.1, valinit=linear_density, valstep=0.001)
	s_tmin = Slider(ax_tmin, "T min (N)", 0.1, 200.0, valinit=t_min, valstep=0.1)
	s_tmax = Slider(ax_tmax, "T max (N)", 0.2, 300.0, valinit=t_max, valstep=0.1)

	def update(_val: float) -> None:
		# Ensure min < max for a sensible range. Avoid recursion with set_val by skipping if unchanged.
		current_tmin = max(0.001, s_tmin.val)
		current_tmax = max(current_tmin + 0.001, s_tmax.val)
		if current_tmin != s_tmin.val:
			s_tmin.set_val(current_tmin)
		if current_tmax != s_tmax.val:
			s_tmax.set_val(current_tmax)

		mode_int = int(round(s_mode.val))
		length_val = s_length.val
		rho_val = s_rho.val

		tension_vals = np.linspace(current_tmin, current_tmax, num_points)
		freq_vals = frequency_from_tension(tension_vals, mode=mode_int, length_m=length_val, linear_density=rho_val)
		line.set_xdata(tension_vals)
		line.set_ydata(freq_vals)
		ax.relim()
		ax.autoscale_view()
		fig.canvas.draw_idle()

	s_mode.on_changed(update)
	s_length.on_changed(update)
	s_rho.on_changed(update)
	s_tmin.on_changed(update)
	s_tmax.on_changed(update)

	# Reset button for convenience
	reset_ax = plt.axes([0.82, 0.05, 0.13, 0.05])
	btn_reset = Button(reset_ax, "Reset", color=ax_color, hovercolor="0.9")

	def reset(_event: object) -> None:
		s_mode.reset()
		s_length.reset()
		s_rho.reset()
		s_tmin.reset()
		s_tmax.reset()

	btn_reset.on_clicked(reset)

	plt.show()


def animate_standing_wave(
	mode: int = 1,
	length_m: float = 1.0,
	linear_density: float = 0.01,
	tension: float = 20.0,
	amplitude: float = 0.02,
	fps: int = 30,
	duration_s: float = 5.0,
	save_path: str | None = None,
) -> None:
	"""Animate the standing wave displacement over time."""
	if tension <= 0 or length_m <= 0 or linear_density <= 0 or mode < 1:
		raise ValueError("All parameters must be positive and mode >= 1.")

	freq = frequency_from_tension(np.array([tension]), mode, length_m, linear_density)[0]
	wave_speed = np.sqrt(tension / linear_density)
	wavelength = 2.0 * length_m / mode
	wavenumber = 2.0 * np.pi / wavelength  # k
	omega = 2.0 * np.pi * freq

	x_vals = np.linspace(0.0, length_m, 400)
	y_base = np.sin(wavenumber * x_vals)  # spatial part for fixed ends

	fig, ax = plt.subplots(figsize=(8, 4))
	[line] = ax.plot(x_vals, amplitude * y_base, color="tab:red", lw=2)
	ax.set_ylim(-1.2 * amplitude, 1.2 * amplitude)
	ax.set_xlim(0, length_m)
	ax.set_xlabel("Position x (m)")
	ax.set_ylabel("Transverse displacement y (m)")
	ax.grid(True, ls="--", alpha=0.4)
	fig.suptitle(f"Standing wave animation: n={mode}, f={freq:.2f} Hz, T={tension} N")

	frames = int(fps * duration_s)

	def init() -> tuple[object]:
		line.set_ydata(amplitude * y_base)
		return (line,)

	def update(frame_idx: int) -> tuple[object]:
		t = frame_idx / fps
		y_t = amplitude * y_base * np.cos(omega * t)
		line.set_ydata(y_t)
		return (line,)

	anim = FuncAnimation(fig, update, init_func=init, frames=frames, interval=1000 / fps, blit=True)

	if save_path:
		# Saving requires ffmpeg installed; skip if unavailable.
		try:
			anim.save(save_path, fps=fps)
		except Exception as exc:  # pragma: no cover - runtime convenience
			print(f"Could not save animation: {exc}")

	plt.show()


def interactive_animation(
	mode: int = 1,
	length_m: float = 1.0,
	linear_density: float = 0.01,
	tension: float = 20.0,
	amplitude: float = 0.02,
	fps: int = 120,
	n_points: int = 400,
) -> None:
	"""Standing wave animation with sliders to tweak parameters in real time."""

	state = {
		"mode": float(mode),
		"length": float(length_m),
		"rho": float(linear_density),
		"tension": float(tension),
		"amplitude": float(amplitude),
	}

	def derived() -> dict[str, float]:
		freq_val = frequency_from_tension(np.array([state["tension"]]), int(round(state["mode"])), state["length"], state["rho"])[0]
		wavelength = 2.0 * state["length"] / int(round(state["mode"]))
		k_val = 2.0 * np.pi / wavelength
		omega_val = 2.0 * np.pi * freq_val
		return {"freq": freq_val, "k": k_val, "omega": omega_val}

	params = derived()
	x_vals = np.linspace(0.0, state["length"], n_points)
	y_base = np.sin(params["k"] * x_vals)

	fig, ax = plt.subplots(figsize=(8, 4))
	plt.subplots_adjust(left=0.1, bottom=0.30)
	[line] = ax.plot(x_vals, state["amplitude"] * y_base, color="tab:red", lw=2)
	ax.set_xlim(0, state["length"])
	ax.set_ylim(-1.5 * state["amplitude"], 1.5 * state["amplitude"] if state["amplitude"] > 0 else 0.05)
	ax.set_xlabel("Position x (m)")
	ax.set_ylabel("Transverse displacement y (m)")
	text_info = ax.text(0.02, 0.9, f"f = {params['freq']:.2f} Hz", transform=ax.transAxes)
	text_pause = ax.text(0.02, 0.84, "", transform=ax.transAxes, color="tab:purple")
	ax.grid(True, ls="--", alpha=0.4)
	fig.suptitle("Standing wave animation with sliders")

	frames = 5 * fps  # loop a few seconds and repeat
	last_frame_idx = 0
	paused = {"flag": False}

	def update_frame(frame_idx: int) -> tuple[object]:
		nonlocal last_frame_idx
		t = frame_idx / fps
		y_t = state["amplitude"] * np.sin(params["k"] * x_vals) * np.cos(params["omega"] * t)
		line.set_data(x_vals, y_t)
		last_frame_idx = frame_idx
		return (line,)

	anim = FuncAnimation(fig, update_frame, frames=frames, interval=1000 / fps, blit=True)

	ax_color = "lightgoldenrodyellow"
	ax_mode = plt.axes([0.10, 0.22, 0.35, 0.03], facecolor=ax_color)
	ax_len = plt.axes([0.55, 0.22, 0.35, 0.03], facecolor=ax_color)
	ax_rho = plt.axes([0.10, 0.17, 0.35, 0.03], facecolor=ax_color)
	ax_tension = plt.axes([0.55, 0.17, 0.35, 0.03], facecolor=ax_color)
	ax_amp = plt.axes([0.10, 0.12, 0.35, 0.03], facecolor=ax_color)

	s_mode = Slider(ax_mode, "n (mode)", 1, 10, valinit=state["mode"], valstep=1)
	s_len = Slider(ax_len, "length l (m)", 0.2, 5.0, valinit=state["length"], valstep=0.01)
	s_rho = Slider(ax_rho, "rho (kg/m)", 0.001, 0.1, valinit=state["rho"], valstep=0.001)
	s_tension = Slider(ax_tension, "Tension T (N)", 0.5, 200.0, valinit=state["tension"], valstep=0.1)
	s_amp = Slider(ax_amp, "Amplitude A (m)", 0.001, 0.05, valinit=state["amplitude"], valstep=0.001)

	def on_change(_val: float) -> None:
		nonlocal x_vals, y_base
		state["mode"] = s_mode.val
		state["length"] = s_len.val
		state["rho"] = s_rho.val
		state["tension"] = s_tension.val
		state["amplitude"] = s_amp.val

		params.update(derived())
		n_int = int(round(state["mode"]))
		x_vals = np.linspace(0.0, state["length"], n_points)
		y_base = np.sin(params["k"] * x_vals)

		line.set_data(x_vals, state["amplitude"] * y_base)
		ax.set_xlim(0, state["length"])
		ax.set_ylim(-1.5 * state["amplitude"], 1.5 * state["amplitude"] if state["amplitude"] > 0 else 0.05)
		text_info.set_text(f"f = {params['freq']:.2f} Hz, n = {n_int}")
		ax.figure.canvas.draw_idle()

	s_mode.on_changed(on_change)
	s_len.on_changed(on_change)
	s_rho.on_changed(on_change)
	s_tension.on_changed(on_change)
	s_amp.on_changed(on_change)

	reset_ax = plt.axes([0.82, 0.05, 0.13, 0.05])
	btn_reset = Button(reset_ax, "Reset", color=ax_color, hovercolor="0.9")
	stop_ax = plt.axes([0.64, 0.05, 0.15, 0.05])
	btn_stop = Button(stop_ax, "Stop", color=ax_color, hovercolor="0.9")

	def reset(_event: object) -> None:
		s_mode.reset()
		s_len.reset()
		s_rho.reset()
		s_tension.reset()
		s_amp.reset()
		text_pause.set_text("")
		if paused["flag"]:
			anim.event_source.start()
			paused["flag"] = False
			btn_stop.label.set_text("Stop")

	btn_reset.on_clicked(reset)

	def toggle_pause(_event: object) -> None:
		if paused["flag"]:
			anim.event_source.start()
			paused["flag"] = False
			btn_stop.label.set_text("Stop")
			text_pause.set_text("")
		else:
			anim.event_source.stop()
			paused["flag"] = True
			btn_stop.label.set_text("Resume")
			t_now = last_frame_idx / fps
			n_int = int(round(state["mode"]))
			text_pause.set_text(
				f"Paused: t={t_now:.2f}s, f={params['freq']:.2f}Hz, n={n_int}, T={state['tension']:.3f}N"
			)

	btn_stop.on_clicked(toggle_pause)

	plt.show()


def sample_values(mode: int, length_m: float, linear_density: float, tensions: list[float]) -> None:
	"""Print a small table of frequencies for specific tensions."""
	lam = wavelength(mode, length_m)
	print(f"Mode n = {mode}, length = {length_m} m, linear density = {linear_density} kg/m")
	print(f"Wavelength lambda = {lam:.4f} m")
	print("Tension (N)    Frequency (Hz)")
	for t in tensions:
		f_val = frequency_from_tension(np.array([t]), mode, length_m, linear_density)[0]
		print(f"{t:11.3f}    {f_val:13.3f}")


if __name__ == "__main__":
	# Adjust parameters here to match your experiment setup.
	DEFAULT_MODE = 1
	DEFAULT_LENGTH_M = 1.0  # meters
	DEFAULT_LINEAR_DENSITY = 0.010  # kg/m

	# sample_values(
	# 	mode=DEFAULT_MODE,
	# 	length_m=DEFAULT_LENGTH_M,
	# 	linear_density=DEFAULT_LINEAR_DENSITY,
	# 	tensions=[1, 5, 10, 20, 40],
	# )

	# demo_plot(
	# 	t_min=1.0,
	# 	t_max=50.0,
	# 	num_points=300,
	# 	mode=DEFAULT_MODE,
	# 	length_m=DEFAULT_LENGTH_M,
	# 	linear_density=DEFAULT_LINEAR_DENSITY,
	# )

	# interactive_plot(
	# 	t_min=1.0,
	# 	t_max=50.0,
	# 	num_points=300,
	# 	mode=DEFAULT_MODE,
	# 	length_m=DEFAULT_LENGTH_M,
	# 	linear_density=DEFAULT_LINEAR_DENSITY,
	# )

	# animate_standing_wave(
	# 	mode=DEFAULT_MODE,
	# 	length_m=DEFAULT_LENGTH_M,
	# 	linear_density=DEFAULT_LINEAR_DENSITY,
	# 	tension=20.0,
	# 	amplitude=0.02,
	# 	fps=30,
	# 	duration_s=5.0,
	# 	# save_path="standing_wave.mp4",
	# )

	# Interactive animation with sliders (combined view) - keep this
	interactive_animation(
		mode=DEFAULT_MODE,
		length_m=DEFAULT_LENGTH_M,
		linear_density=DEFAULT_LINEAR_DENSITY,
		tension=20.0,
		amplitude=0.02,
		fps=30,
		n_points=400,
	)
