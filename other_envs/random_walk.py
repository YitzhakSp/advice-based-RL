

#random walk with double states
class RWEnv():

    def __init__(self,
                  length,
                  first_double_state,
                  n_double_states):
        self.length=length
        last_double_state = first_double_state + n_double_states - 1
        assert ((first_double_state > 2) and (last_double_state < length - 3))
        assert n_double_states > 0
        RW_graph = {}
        state_types=[None for i in range(length)]
        for i in range(length):
            if i==0 or i==(length-1):
                state_types[i]=['g']
            elif i>=first_double_state and i<=last_double_state:
                state_types[i]=['top','bot']
            else:
                state_types[i]=['sng']
        RW_graph={}
        RW_graph[(0, 'g')], RW_graph[(length - 1, 'g')] = [(0, 'g')], [(length - 1, 'g')]
        for i in range(1,length-1):
            if len(state_types[i])==1: #single state
                RW_graph[(i, 'sng')]=[]
                for state_type in state_types[i-1]:
                    RW_graph[(i, 'sng')].append((i-1,state_type))
                for state_type in state_types[i+1]:
                    RW_graph[(i, 'sng')].append((i+1,state_type))
            else : # double state
                RW_graph[(i, 'top')] = []
                RW_graph[(i, 'bot')] = []
                for state_type in state_types[i + 1]:
                    RW_graph[(i, 'top')].append((i + 1, state_type))
                    RW_graph[(i, 'bot')].append((i + 1, state_type))
        self.RW_graph = RW_graph

    def get_actions(self,s):
        return self.RW_graph[s]

    def set_state(self,s):
        assert s in self.RW_graph
        self.state=s

    def step(self,a):
        assert a in self.RW_graph[self.state]
        self.state=a
        r=-1
        done=True if self.state[1]=='g' else False
        return self.state,r,done

    def get_actions(self,s):
        return self.RW_graph[s]

#attention function
def set_h_rw(RW_graph,length):
    h={}
    for key in RW_graph:
        if key[1]=='sng':
            h[key]=1
        else:
            h[key]=1
    return h

#for random walk
def s_to_str(s):
    return str(s[0])+'_'+s[1]

def str_to_s_rw(s_str):
    spl=s_str.split('_')
    return  (int(spl[0]),spl[1])

def a_to_str(a):
    return str(a[0])+'_'+a[1]

def str_to_a_rw(a_str):
    spl=a_str.split('_')
    return  (int(spl[0]),spl[1])

def str_to_sa_rw(sa_str):
    spl=sa_str.split('_')
    s=(int(spl[0]),spl[1])
    a=(int(spl[2]),spl[3])
    return s,a

#for random walk
def sa_to_str_rw(s,a):
    return str(s[0])+'_'+s[1]+'_'+str(a[0])+'_'+a[1]
