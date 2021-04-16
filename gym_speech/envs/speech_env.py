import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
from glob import glob
from random import sample
import librosa

class SpeechEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, obs_dim=12, data_path='./', sr=16000, win_length=400, hop_length=160):
        self.sr = sr
        self.n_fft = win_length
        self.hop_length = hop_length
        
        self.observation_space = spaces.Box(low=-10, high=10, shape=(13,))

        low = np.array([0.0, -6.0, -0.5, -7.0, -1.0, -2.0, 0.0, -0.1, -3.0, -3.0, 1.5, -3.0, -3.0, -3.0, -4.0, -6.0, 0.0, 0.0, -1.0])
        high = np.array([1.0, -3.5, 0.0, 0.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 5.5, 2.5, 4.0, 5.0, 2.0, 0.0, 1.0, 1.0, 1.0])
        L = len(low)
        self.action_space = spaces.Box(low=low, high=high, shape=(L,))

        self.path = glob(os.path.join(data_path, '*.wav'))
        
        #self.i_path = 0
        self.reset()

    def step(self, action):
        reward = self.get_reward(action)
        ob, episode_over = self.next_state()
        return ob, reward, episode_over, {}
    
    def next_state(self):
        self.i_feature += 1
        if self.i_feature >= self.L_feature - 1:
            episode_over = True
        else:
            episode_over = False
        ob = self.feature_sequences[self.i_feature]
        return ob, episode_over

    def get_reward(self, action):
        return 0

    def reset(self):
        audio, _ = librosa.load(sample(self.path,1)[0], sr=self.sr)
        self.feature_sequences = self.feature_extract(audio) # tbins x 13
        self.i_feature = 0
        self.L_feature = len(self.feature_sequences)
        
        return self.feature_sequences[self.i_feature]

    def render(self):
        pass
                
    def feature_extract(self, y, n_mfcc=13, n_mels=40, fmin=0, fmax=None):
        # mfcc: 13 x (L//hop + 1)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_fft=self.n_fft,
                                    n_mfcc=n_mfcc, n_mels=n_mels,
                                    hop_length=self.hop_length,
                                    fmin=fmin, fmax=fmax, htk=False)
        return mfcc.T, L


