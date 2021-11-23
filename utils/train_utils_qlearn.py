def evaluate(agent,env,max_episode_steps,gamma):
    steps=0
    score=0
    dfact=1.0
    terminal=False
    s=env.reset()
    eps_eval=0
    while (not terminal) and (steps<max_episode_steps):
        a=agent.choose_action_from_qtab(s,eps_eval)
        s_, r, terminal = env.step(a)
        agent.add_state_to_qtab(s_)
        score+= dfact * r
        agent.learn(s, a, r, s_, terminal)
        s=s_
        steps+=1
        dfact*=gamma
    score=round(score,3)
    return score
