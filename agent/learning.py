import settings

def get_loss(observation, next_observation, reward, termination, agent):
    print("Agent:", agent)
    print("Reward:", reward)
    print("Termination:", termination)

    if termination:
        y = reward
    else:
        y = reward + settings.DISCOUNT * agent.target_evaluate(next_observation)
    
    print("Y:", y)
    print("Evaluate:", agent.current_evaluate(observation))
    return (y - agent.current_evaluate(observation)) ** 2