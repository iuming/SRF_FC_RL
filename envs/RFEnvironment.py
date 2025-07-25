import numpy as np
import gymnasium as gym
from gymnasium import spaces
from llrflibs.rf_sim import *
from llrflibs.rf_control import *
import matplotlib.pyplot as plt


class RFEnvironment(gym.Env):
    def __init__(self):
        # General parameters
        self.Ts = 1e-6
        self.t_fill = 510
        self.t_flat = 1300

        # RF source
        self.fsrc = -460
        self.Asrc = 1
        self.pha_src = 0

        # I/Q modulator
        self.pulsed = False
        self.buf_size = 2048 * 8
        self.base_pul = np.zeros(self.buf_size, dtype=complex)
        self.base_cw = 1
        self.base_pul[:self.t_flat] = 1.0
        self.buf_id = 0

        # amplifier
        self.gain_dB = 20 * np.log10(12e6)

        # cavity
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
        self.Ad = Ad
        self.Bd = Bd
        self.Cd = Cd
        self.Dd = Dd

        self.state_m = np.matrix(np.zeros(Bd.shape))
        self.state_vc = 0.0

        self.sim_len = 2048 * 16
        self.pul_len = 2048 * 10

        self.current_step = 0

        # 定义动作空间和状态空间
        self.action_space = spaces.Box(low=0.8, high=1.2, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # 用于记录数据
        self.sig_src_real = []
        self.sig_src_imag = []
        self.sig_amp_real = []
        self.sig_amp_imag = []
        self.sig_vc_real = []
        self.sig_vc_imag = []
        self.sig_dw = []
        self.action_history = []

    def sim_rfsrc(self):
        pha = self.pha_src + 2.0 * np.pi * self.fsrc * self.Ts
        S0 = self.Asrc * np.exp(1j * pha)
        self.pha_src = pha
        return S0

    def sim_iqmod(self, sig_in):
        if self.pulsed:
            sig_out = sig_in * self.base_pul[self.buf_id if self.buf_id < len(self.base_pul) else -1]
        else:
            sig_out = sig_in * self.base_cw
        return sig_out

    def sim_amp(self, sig_in):
        return sig_in * 10.0 ** (self.gain_dB / 20.0)

    def sim_cav(self, vf_step):
        if self.pulsed:
            vb = -self.RL * self.beam_pul[self.buf_id if self.buf_id < len(self.beam_pul) else -1]
        else:
            vb = self.beam_cw

        dw_micr = 2.0 * np.pi * np.random.randn() * 10
        status, vc, vr, dw, state_m = sim_scav_step(self.wh,
                                                    0,
                                                    self.dw0 + dw_micr,
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
        self.state_m = state_m
        return vc, vr, dw

    def step(self, action):
        # record action
        self.action_history.append(action[0])

        # 假设动作影响 RF 源的幅度
        self.Asrc = action[0]

        # RF signal source
        S0 = self.sim_rfsrc()

        # emulate the pulse
        if self.pulsed:
            self.buf_id += 1
            if self.buf_id >= self.pul_len:
                self.buf_id = 0

        # I/Q modulator
        S1 = self.sim_iqmod(S0)

        # amplifier
        S2 = self.sim_amp(S1)

        # cavity
        vc, vr, dw = self.sim_cav(S2)

        # 构建观测值
        observation = np.array([np.real(S0), np.imag(S0), np.real(S2), np.imag(S2), np.real(vc), np.imag(vc)],
                               dtype=np.float32)

        # 简单的奖励函数，可根据实际需求修改
        reward = -np.abs(np.abs(vc) * 1e-6 - 10)  # 目标是使腔体电压接近10 MV

        terminated = self.current_step >= self.sim_len - 1
        truncated = False
        info = {}

        # 记录数据
        self.sig_src_real.append(np.real(S0))
        self.sig_src_imag.append(np.imag(S0))
        self.sig_amp_real.append(np.real(S2))
        self.sig_amp_imag.append(np.imag(S2))
        self.sig_vc_real.append(np.real(vc))
        self.sig_vc_imag.append(np.imag(vc))
        # 将 dw 转换为一维数组
        self.sig_dw.append(np.array(dw).flatten()[0])

        self.current_step += 1

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 重置状态
        self.pha_src = 0
        self.buf_id = 0
        self.state_m = np.matrix(np.zeros(self.Bd.shape))
        self.state_vc = 0.0
        self.current_step = 0

        # 清空记录的数据
        self.sig_src_real = []
        self.sig_src_imag = []
        self.sig_amp_real = []
        self.sig_amp_imag = []
        self.sig_vc_real = []
        self.sig_vc_imag = []
        self.sig_dw = []
        self.action_history = []

        # 初始观测值
        S0 = self.sim_rfsrc()
        S1 = self.sim_iqmod(S0)
        S2 = self.sim_amp(S1)
        vc, vr, dw = self.sim_cav(S2)
        observation = np.array([np.real(S0), np.imag(S0), np.real(S2), np.imag(S2), np.real(vc), np.imag(vc)],
                               dtype=np.float32)

        info = {}
        return observation, info

    def render(self, mode='human'):
        if mode == 'human':
            plt.figure()
            plt.plot(self.sig_src_real, label='Real(Src)')
            plt.plot(self.sig_src_imag, '--', label='Imag(Src)')
            plt.plot(self.sig_amp_real, label='Real(Amp)')
            plt.plot(self.sig_amp_imag, '--', label='Imag(Amp)')
            plt.plot(self.sig_vc_real, label='Real(Vc)')
            plt.plot(self.sig_vc_imag, '--', label='Imag(Vc)')
            plt.xlabel('Time Step')
            plt.ylabel('Amplitude')
            plt.title('RF System Signals')
            plt.legend()
            plt.show()

            plt.figure()
            sig_vc = np.array(self.sig_vc_real) + 1j * np.array(self.sig_vc_imag)
            plt.subplot(3, 1, 1)
            plt.plot(np.abs(sig_vc) * 1e-6)
            plt.xlabel('Time (Ts)')
            plt.ylabel('Cavity Voltage (MV)')
            plt.subplot(3, 1, 2)
            plt.plot(np.angle(sig_vc) * 180 / np.pi)
            plt.xlabel('Time (Ts)')
            plt.ylabel('Cavity Phase (deg)')
            plt.subplot(3, 1, 3)
            plt.plot(np.array(self.sig_dw) / 2 / np.pi)
            plt.xlabel('Time (Ts)')
            plt.ylabel('Detuning (Hz)')
            plt.show()

            plt.figure()
            plt.plot(self.action_history, label='Action')
            plt.xlabel('Time Step')
            plt.ylabel('Action Value')
            plt.title('Action History')
            plt.legend()
            plt.show()