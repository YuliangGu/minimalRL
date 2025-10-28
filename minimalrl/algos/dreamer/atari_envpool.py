import envpool

env_id = "Pong-v5"
envs = envpool.make(
        env_id,
        env_type="gym",
        num_envs=16,
        repeat_action_probability=0.25,
        clip_rewards=False,
    )

# print all envs properties and info
print(envs)