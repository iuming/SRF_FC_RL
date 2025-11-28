import numpy as np
import math
import socket
import time
import gymnasium as gym
from gymnasium import spaces
try:
    from epics import PV as EpicsPV
except Exception:
    EpicsPV = None
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:
    PPO = None

PV = {
    "probe_amp": "LLRF:TEST:Probe:Amplitude",
    "probe_phase": "LLRF:TEST:Probe:Phase",
    "forward_amp": "LLRF:TEST:Forward:Amplitude",
    "forward_phase": "LLRF:TEST:Forward:Phase",
    "reflected_amp": "LLRF:TEST:Reflected:Amplitude",
    "reflected_phase": "LLRF:TEST:Reflected:Phase",
    "piezo_drive": "LLRF:TEST:Piezo:Drive"
}

class PVClient:
    def __init__(self, pv_map, use_real_cavity=False, cavity_points=2048):
        self.pv_map = pv_map
        self.use_real_cavity = use_real_cavity
        self.cavity_points = cavity_points
        self._store = {}
        if EpicsPV is not None:
            self._pvs = {}
            for k, v in pv_map.items():
                try:
                    self._pvs[k] = EpicsPV(v)
                except Exception:
                    self._pvs[k] = None
        else:
            self._pvs = {}

    def get(self, key):
        if self.use_real_cavity and EpicsPV is not None and key in self._pvs and self._pvs[key] is not None:
            val = self._pvs[key].get()
            if val is not None:
                return val
        return self._store.get(key, None)

    def put(self, key, value):
        if self.use_real_cavity and EpicsPV is not None and key in self._pvs and self._pvs[key] is not None:
            try:
                self._pvs[key].put(value)
                return
            except Exception:
                pass
        self._store[key] = value

class VirtualCavity:
    def __init__(self, points=2048):
        self.points = points
        self.time = np.linspace(0.0, (points-1)*1e-6, points)
        self.omega0 = 2*math.pi*1.3e9
        self.QL = 1e7

    def generate_waveforms(self):
        t = self.time
        probe_amp = 1.0 + 0.01*np.sin(2*np.pi*50*t)
        probe_phase = 0.1*np.sin(2*np.pi*30*t)
        forward_amp = 1.0 + 0.01*np.sin(2*np.pi*50*t + 0.1)
        forward_phase = 0.1*np.sin(2*np.pi*30*t + 0.05)
        reflected_amp = 0.05*np.ones_like(t)
        reflected_phase = 0.0*np.ones_like(t)
        return probe_amp, probe_phase, forward_amp, forward_phase, reflected_amp, reflected_phase

class Keysight33600A:
    def __init__(self, ip: str, port: int = 5025, timeout: float = 2.0):
        self.ip = ip
        self.port = port
        self.timeout = timeout

    def _send_scpi(self, cmd: str):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        resp = b""
        try:
            s.connect((self.ip, self.port))
            s.sendall(cmd.encode("ascii") + b"\n")
            while True:
                try:
                    part = s.recv(4096)
                except socket.timeout:
                    break
                if not part:
                    break
                resp += part
        except Exception:
            resp = b""
        finally:
            try:
                s.close()
            except Exception:
                pass
        return resp

    def send_waveform_ascii(self, channel: int, waveform: np.ndarray):
        wf = np.clip(np.asarray(waveform, dtype=float), 0.0, 3.0)
        chunk_size = 512
        name = f"ARB_CH{channel}"
        try:
            self._send_scpi(f"SOUR{channel}:DATA:ARB:DEF {name}")
            for i in range(0, len(wf), chunk_size):
                chunk = wf[i:i+chunk_size]
                data_str = ",".join([f"{v:.6f}" for v in chunk])
                self._send_scpi(f"SOUR{channel}:DATA:ARB:APPEND {name},{data_str}")
            self._send_scpi(f"SOUR{channel}:FUNC:ARB {name}")
            self._send_scpi(f"OUTP{channel} ON")
        except Exception:
            pass

class KalmanFilter:
    def __init__(self, size, q=1e-3, r=1e-2):
        self.size = int(size)
        self.q = float(q)
        self.r = float(r)
        self.x = np.zeros(self.size)
        self.p = np.ones(self.size) * 1.0

    def update(self, z):
        z = np.asarray(z, dtype=float)
        if z.shape != self.x.shape:
            z = np.resize(z, self.size)
        self.p = self.p + self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1 - k) * self.p
        return self.x

class PIController:
    def __init__(self, size, kp=1e-3, ki=1e-4, dt=1e-3, out_min=0.0, out_max=3.0):
        self.size = int(size)
        self.kp = float(kp)
        self.ki = float(ki)
        self.dt = float(dt)
        self.integral = np.zeros(self.size)
        self.out_min = float(out_min)
        self.out_max = float(out_max)

    def update(self, error, feedforward=None):
        e = np.asarray(error, dtype=float)
        if e.shape != (self.size,):
            e = np.resize(e, self.size)
        self.integral += e * self.dt
        out = self.kp * e + self.ki * self.integral
        if feedforward is not None:
            ff = np.asarray(feedforward, dtype=float)
            if ff.shape != (self.size,):
                ff = np.resize(ff, self.size)
            out = out + ff
        out = np.clip(out, self.out_min, self.out_max)
        return out

class CavityController:
    def __init__(self, pv_map=PV, use_real_cavity=False, keysight_ip="10.4.169.196", channel=1, points=2048):
        self.pv = PVClient(pv_map, use_real_cavity, points)
        self.use_real_cavity = use_real_cavity
        self.points = points
        self.time = np.linspace(0.0, (points-1)*1e-6, points)
        self.use_keysight = use_real_cavity
        self.keysight = Keysight33600A(keysight_ip) if self.use_keysight else None
        self.channel = channel
        self.kalman = KalmanFilter(points, q=1e-4, r=1e-3)
        self.pi = PIController(points, kp=5e-4, ki=1e-5, dt=self.time[1]-self.time[0], out_min=0.0, out_max=3.0)
        self.omega0 = 2.0*math.pi*1.3e9
        self.QL = 1e7
        self.sim_cavity = VirtualCavity(points)

    def _get_wave(self, key):
        val = self.pv.get(key)
        if val is not None:
            a = np.asarray(val, dtype=float)
            if a.size == self.points:
                return a
            if a.size == 1:
                return np.full(self.points, a.item(), dtype=float)
            return np.resize(a, self.points)
        probe_amp, probe_phase, fwd_amp, fwd_phase, ref_amp, ref_phase = self.sim_cavity.generate_waveforms()
        self.pv.put("probe_amp", probe_amp)
        self.pv.put("probe_phase", probe_phase)
        self.pv.put("forward_amp", fwd_amp)
        self.pv.put("forward_phase", fwd_phase)
        self.pv.put("reflected_amp", ref_amp)
        self.pv.put("reflected_phase", ref_phase)
        return {"probe_amp": probe_amp, "probe_phase": probe_phase, 
                "forward_amp": fwd_amp, "forward_phase": fwd_phase, 
                "reflected_amp": ref_amp, "reflected_phase": ref_phase}[key]

    def compute_detuning(self, cav_amp, cav_phase, fwd_amp, fwd_phase):
        dt = self.time[1]-self.time[0]
        Vc = cav_amp * np.exp(1j*np.radians(cav_phase))
        Vf = fwd_amp * np.exp(1j*np.radians(fwd_phase))
        dVc_dt = np.gradient(Vc, dt)
        gamma = self.omega0 / (2.0*self.QL)
        X = dVc_dt + gamma*Vc - Vf
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = X / (Vc + 1e-12)
            delta_omega = np.imag(ratio)
            det_Hz = delta_omega / (2.0*math.pi)
            det_Hz = np.nan_to_num(det_Hz, nan=0.0, posinf=0.0, neginf=0.0)
        return det_Hz

    def step(self, action=None):
        if action is not None:
            action = np.clip(action, 0.0, 3.0)
            if self.keysight is not None:
                self.keysight.send_waveform_ascii(self.channel, action)
            self.pv.put("piezo_drive", action.copy())
        if self.use_real_cavity:
            cav_amp = self._get_wave("probe_amp")
            cav_phase = self._get_wave("probe_phase")
            fwd_amp = self._get_wave("forward_amp")
            fwd_phase = self._get_wave("forward_phase")
            ref_amp = self._get_wave("reflected_amp")
            ref_phase = self._get_wave("reflected_phase")
        else:
            waves = self.sim_cavity.generate_waveforms()
            cav_amp, cav_phase, fwd_amp, fwd_phase, ref_amp, ref_phase = waves
            self.pv.put("probe_amp", cav_amp)
            self.pv.put("probe_phase", cav_phase)
            self.pv.put("forward_amp", fwd_amp)
            self.pv.put("forward_phase", fwd_phase)
            self.pv.put("reflected_amp", ref_amp)
            self.pv.put("reflected_phase", ref_phase)
        det = self.compute_detuning(cav_amp, cav_phase, fwd_amp, fwd_phase)
        det_filtered = self.kalman.update(det)
        error = -det_filtered
        piezo_wf = self.pi.update(error)
        if self.keysight is not None:
            self.keysight.send_waveform_ascii(self.channel, piezo_wf)
        self.pv.put("piezo_drive", piezo_wf.copy())
        obs = np.vstack([cav_amp, cav_phase, fwd_amp, fwd_phase, ref_amp, ref_phase]).astype(np.float32)
        reward = -float(np.mean(np.abs(det)))
        return obs, reward, det, det_filtered

class PiezoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, controller: CavityController, episode_length=5):
        super().__init__()
        self.controller = controller
        self.fs = controller.points
        self.episode_length = episode_length
        self.current_step = 0
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6, self.fs), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 3.0, shape=(self.fs,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.controller.pv.put("piezo_drive", np.zeros(self.fs))
        obs, _, _, _ = self.controller.step()
        self.current_step = 0
        return obs, {}

    def step(self, action):
        obs, reward, det, det_f = self.controller.step(action)
        self.current_step += 1
        done = self.current_step >= self.episode_length
        info = {"det": det, "det_filtered": det_f}
        return obs, reward, done, False, info

    def render(self):
        det_mean = np.mean(self.controller.compute_detuning(
            self.controller._get_wave("probe_amp"),
            self.controller._get_wave("probe_phase"),
            self.controller._get_wave("forward_amp"),
            self.controller._get_wave("forward_phase")))
        print(f"Mean detuning: {det_mean:.6f} Hz")

class SaveBestModelCallback(BaseCallback):
    def __init__(self, check_freq=1, verbose=1, save_path="best_model.zip"):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_reward = -np.inf

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.locals.get("rewards", [0]))
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.model.save(self.save_path)
        return True

if __name__ == "__main__":
    rc = CavityController(use_real_cavity=False)
    env = PiezoEnv(rc, episode_length=5)
    if PPO is not None:
        model = PPO("MlpPolicy", env, verbose=1)
        callback = SaveBestModelCallback(check_freq=1)
        model.learn(total_timesteps=10000, callback=callback)
        model.save("final_model.zip")
    else:
        print("stable-baselines3 not available, skipping RL training.")
