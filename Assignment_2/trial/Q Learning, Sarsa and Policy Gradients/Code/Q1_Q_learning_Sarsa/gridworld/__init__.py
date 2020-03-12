#https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym
from gym.envs.registration import register

register(
    id='GridWorld-v0',
    entry_point='gridworld.envs:GridWorld',
)
