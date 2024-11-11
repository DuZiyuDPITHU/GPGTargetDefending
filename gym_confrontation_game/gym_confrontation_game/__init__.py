from gym.envs.registration import register

register(
    id='maconfrontation-v0',
    entry_point='gym_confrontation_game.envs:ConfrontationEnv',
    max_episode_steps=500,
)
