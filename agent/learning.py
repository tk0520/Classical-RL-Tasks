import settings

def q_value(observation, reward, termination, agent):
    print("Agent:", agent)
    print("Reward:", reward)
    print("Termination:", termination)

    if termination:
        y = reward
    else:
        y = reward + settings.DISCOUNT * agent.forward(observation)
    
    print("Y:", y)
