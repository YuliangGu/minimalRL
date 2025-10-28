import torch
import numpy as np
import gym

is_legacy_gym = gym.__version__ < "0.26.0"

from minimalrl.core.envs import make_vector_env

def test_vector_env_shapes():
    env_id = "CartPole-v1" # observation space shape (4,), action space shape (1,)
    num_envs = 50
    seed = 42

    vec_env = make_vector_env(env_id, seed, num_envs, backend="envpool")

    obs = vec_env.reset() # should be (num_envs, obs_dim)
    
    assert obs.shape == (num_envs, 4)



    act_num = vec_env.action_space.n
    act = np.random.randint(0, act_num, size=(num_envs,))

    if is_legacy_gym:
        next_obs, rewards, term, infos = vec_env.step(act)
    else:
        next_obs, rewards, term, trunc, infos = vec_env.step(act)

    assert next_obs.shape == (num_envs, 4)
    assert rewards.shape == (num_envs,)
    assert term.shape == (num_envs,)