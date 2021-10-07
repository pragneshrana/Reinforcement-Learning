#Registering the custom environment 
#Ref: https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym
#Ref: https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai
#main Ref: https://github.com/openai/gym/blob/master/docs/creating-environments.md

from gym.envs.registration import register

register(
    id='GridWorld-v0',
    entry_point='gridworld.envs:GridWorld',
)
