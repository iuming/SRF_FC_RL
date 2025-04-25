import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from llrflibs.rf_sim import *
from llrflibs.rf_control import *


class RFEnvironment(gym.Env):
    def __init__(self, config):
        # 定义动作和观测空间
        self.action_space = gym.spaces.Box(low=-1e4, high=1e4, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_history = [np.zeros(1) for _ in range(3)]

        # 通用参数
        self.Ts = 1e-6
        self.t_fill = config.get("t_fill", 510)
        self.t_flat = config.get("t_flat", 1300)

        # RF 源参数
        self.fsrc = config.get("fsrc", -460)
        self.Asrc = config.get("Asrc", 1)
        self.pha_src = 0

        # I/Q 调制器参数
        self.pulsed = config.get("pulsed", True)
        self.buf_size = config.get("buf_size", 2048 * 8)
        self.base_pul = np.zeros(self.buf_size, dtype=complex)
        self.base_cw = config.get("base_cw", 1)
        self.base_pul[:self.t_flat] = 1.0
        self.buf_id = 0

        # 放大器参数
        self.gain_dB = 20 * np.log10(12e6)

        # 腔参数
        self.mech_modes = {'f': [280, 341, 460, 487, 618],
                           'Q': [40, 20, 50, 80, 100],
                           'K': [2, 0.8, 2, 0.6, 0.2]}
        self.f0 = 1.3e9
        self.beta = 1e4
        self.roQ = 1036
        self.QL = 3e6
        self.RL = 0.5 * self.roQ * self.QL
        self.wh = np.pi * self.f0 / self.QL
        self.ib = 0.008
        self.dw0 = 2 * np.pi * 0

        self.beam_pul = np.zeros(self.buf_size, dtype=complex)
        self.beam_cw = 0
        self.beam_pul[self.t_fill:self.t_flat] = self.ib

        status, Am, Bm, Cm, Dm = cav_ss_mech(self.mech_modes)
        status, Ad, Bd, Cd, Dd, _ = ss_discrete(Am, Bm, Cm, Dm,
                                                Ts=self.Ts,
                                                method='zoh',
                                                plot=False,
                                                plot_pno=10000)
        self.state_m = np.matrix(np.zeros(Bd.shape))
        self.state_vc = 0.0
        self.Ad = Ad
        self.Bd = Bd
        self.Cd = Cd
        self.Dd = Dd

        self.sim_len = 2048 * 500
        self.pul_len = 2048 * 20
        self.dw = 0
        self.prev_dw = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pha_src = 0
        self.buf_id = 0
        self.state_m = np.matrix(np.zeros(self.Bd.shape))
        self.state_vc = 0.0
        self.dw = 0

        S0, self.pha_src = self.sim_rfsrc()
        S1 = self.sim_iqmod(S0)
        S2 = self.sim_amp(S1)
        dw_micr = 2.0 * np.pi * np.random.randn() * 10
        dw_piezo = 0
        vc, vr, self.dw, self.state_vc, self.state_m = self.sim_cav(S2, dw_piezo + dw_micr)

        if isinstance(vc, np.matrix):
            vc = vc.item()

        dw_rate = (self.dw - self.prev_dw) / self.Ts if self.Ts > 0 else 0
        dw_rate = dw_rate[0, 0] if isinstance(dw_rate, np.matrix) else dw_rate
        action_history_array = np.concatenate(self.action_history)
        # action_history_array = action_history_array.flatten()
        action_history_array = np.ravel(action_history_array)


        observation = np.array([np.real(S2), np.imag(S2), np.real(vc), np.imag(vc), np.real(vr), np.imag(vr), dw_rate, *action_history_array], dtype=np.float32)
        info = {}
        return observation, info

    def step(self, action):
        self.action_history.pop(0)
        self.action_history.append(action)
        dw_piezo = 2 * np.pi * action[0]
        prev_dw = self.dw

        S0, self.pha_src = self.sim_rfsrc()

        if self.pulsed:
            self.buf_id += 1
            if self.buf_id >= self.pul_len:
                self.buf_id = 0

        S1 = self.sim_iqmod(S0)
        S2 = self.sim_amp(S1)
        dw_micr = 2.0 * np.pi * np.random.randn() * 10

        vc, vr, self.dw, self.state_vc, self.state_m = self.sim_cav(S2, dw_piezo + dw_micr)


        if isinstance(vc, np.matrix):
            vc = vc.item()

        if isinstance(vr, np.matrix):
            vr = vr.item()

        dw_rate = (self.dw - prev_dw) / self.Ts if self.Ts > 0 else 0
        dw_rate = dw_rate[0, 0] if isinstance(dw_rate, np.matrix) else dw_rate

        try:
            action_history_array = np.concatenate(self.action_history)
            # action_history_array = action_history_array.flatten()
            action_history_array = np.ravel(action_history_array)
            observation = np.array([np.real(S2), np.imag(S2), np.real(vc), np.imag(vc), np.real(vr), np.imag(vr), dw_rate, *action_history_array], dtype=np.float32)
        except ValueError as e:
            print(f"Error creating observation array: {e}")
            print(f"After conversion, vc type: {type(vc)}, shape: {np.shape(vc) if hasattr(vc, 'shape') else 'scalar'}")
            raise
        
        self.dw = float(self.dw)
        base_reward = 2.0 * np.pi * 1000 - np.abs(self.dw)
        # action_magnitude = np.abs(dw_piezo)
        # action_reward = 0.01 * action_magnitude
        # total_reward = base_reward + action_reward
        # reward = total_reward
        dw_improvement = np.abs(prev_dw) - np.abs(self.dw)
        if dw_improvement > 0:
            action_magnitude = np.abs(dw_piezo)
            improvement_reward = 0.01 * action_magnitude * dw_improvement
            reward = float(base_reward + improvement_reward)
        else:
            reward = float(base_reward)
        
        terminated = False
        truncated = False
        info = {
            "state_m": self.state_m.tolist(),
            "state_vc": self.state_vc,
            "dw_micr": dw_micr,
            "base_reward": base_reward,
            "total_reward": reward
        }

        return observation, reward, terminated, truncated, info

    def sim_rfsrc(self):
        pha = self.pha_src + 2.0 * np.pi * self.fsrc * self.Ts
        return self.Asrc * np.exp(1j * pha), pha

    def sim_iqmod(self, sig_in):
        if self.pulsed:
            sig_out = sig_in * self.base_pul[self.buf_id if self.buf_id < len(self.base_pul) else -1]
        else:
            sig_out = sig_in * self.base_cw
        return sig_out

    def sim_amp(self, sig_in):
        return sig_in * 10.0 ** (self.gain_dB / 20.0)

    def sim_cav(self, vf_step, detuning):
        if self.pulsed:
            vb = -self.RL * self.beam_pul[self.buf_id if self.buf_id < len(self.beam_pul) else -1]
        else:
            vb = self.beam_cw

        status, vc, vr, dw, state_m = sim_scav_step(self.wh,
                                                    self.dw,
                                                    detuning,
                                                    vf_step,
                                                    vb,
                                                    self.state_vc,
                                                    self.Ts,
                                                    beta=self.beta,
                                                    state_m0=self.state_m,
                                                    Am=self.Ad,
                                                    Bm=self.Bd,
                                                    Cm=self.Cd,
                                                    Dm=self.Dd,
                                                    mech_exe=True)
        self.state_vc = vc
        return vc, vr, dw, self.state_vc, state_m


# if __name__ == "__main__":
#     env = RFEnvironment()
#     obs, _ = env.reset()
#     for _ in range(100):
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, _ = env.step(action)
#         print(f"Obs: {obs}, Reward: {reward}, Action: {action}")
#         if terminated or truncated:
#             obs, _ = env.reset()
#     env.close()