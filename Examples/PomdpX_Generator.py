class PomdpX:
    """
    A class that handles .pomdpx file generation by writing sections to file.
    """
    def __init__(self, filename, S, A, O):
        self.S = S 
        self.A = A
        self.O = O
        self.file = open(filename, "w+")
        self.template = '''<?xml version="1.0" encoding="ISO-8859-1"?>
        <pomdpx version="0.1" id="rockSample"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:noNamespaceSchemaLocation="pomdpx.xsd">
        <Description> DESC </Description>
        <Discount> DISCOUNT </Discount>
        <Variable> VAR </Variable>
        <InitialStateBelief> INITIAL_B </InitialStateBelief>
        <StateTransitionFunction> STATE_TRANS </StateTransitionFunction>
        <ObsFunction> OBS </ObsFunction>
        <RewardFunction> REWARD </RewardFunction>
        </pomdpx>'''

    def description(self, desc):
        self.template = self.template.replace("DESC", desc)

    def discount(self, γ):
        self.template = self.template.replace("DISCOUNT", str(γ))

    def variable(self):
        var = f'''
        <StateVar vnamePrev="state_prev" vnameCurr="state_curr">
         <NumValues>{len(self.S)}</NumValues>
	    </StateVar>
	    <ObsVar vname="observation">
	     <NumValues>{len(self.O)}</NumValues>
	    </ObsVar>
	    <ActionVar vname="action">
	     <NumValues>{len(self.A)}</NumValues>
	    </ActionVar>
	    <RewardVar vname="reward" />
        '''
        self.template = self.template.replace("VAR", var)
        
    def initial_belief(self, b):
        initial_b = f'''
         <CondProb>
          <Var>state_prev</Var>
          <Parent>null</Parent>
          <Parameter type = "TBL"> 
           <Entry>
            <Instance> - </Instance>
            <ProbTable>{" ".join([f"{p:.20f}" for p in b])}</ProbTable>
           </Entry>
          </Parameter>
         </CondProb>
        '''
        self.template = self.template.replace("INITIAL_B", initial_b)
        
    def state_transition(self, P):        
        state_trans = '''
         <CondProb>
          <Var>state_curr</Var>
          <Parent>action state_prev</Parent>
          <Parameter type = "TBL">'''
         
        for i in range(len(self.S)):          # s'
            for j in range(len(self.S)):      # s
                for k in range(len(self.A)):  # a
                    state_trans += f'''
                    <Entry>
                     <Instance>a{k} s{j} s{i}</Instance>
                     <ProbTable>{P[self.S[j], self.A[k], self.S[i]]}</ProbTable>
                    </Entry>'''          
        
        state_trans += ''' 
          </Parameter>
         </CondProb>
        '''
        
        self.template = self.template.replace("STATE_TRANS", state_trans)
        
    def observation(self, Obs):
        obs = '''
          <CondProb>
           <Var>observation</Var>
           <Parent>action state_curr</Parent>
           <Parameter type = "TBL">'''
           
        for i in range(len(self.O)):          # o 
            for j in range(len(self.S)):      # s'
                for k in range(len(self.A)):  # a
                    obs += f'''
                    <Entry>
                     <Instance>a{k} s{j} o{i}</Instance>
                     <ProbTable>{Obs[self.S[j], self.A[k], self.O[i]]}</ProbTable>
                    </Entry>'''
                    
        obs += ''' 
          </Parameter>
         </CondProb>
        '''
        
        self.template = self.template.replace("OBS", obs)

    def reward(self, R):        
        r = '''
         <Func>
          <Var>reward</Var>
          <Parent>action state_prev state_curr</Parent>
          <Parameter type = "TBL">'''
        
        for i in range(len(self.S)):          # s
            for j in range(len(self.A)):      # a
                for k in range(len(self.S)):  # s'
                    r += f'''
                <Entry>
                 <Instance>a{j} s{i} s{k}</Instance>
                 <ValueTable>{R[self.S[i], self.A[j], self.S[k]]}</ValueTable>
                </Entry>'''                
                
        r += '''
          </Parameter>
         </Func>
        '''
        
        self.template = self.template.replace("REWARD", r)

    def write_to_file(self):
        self.file.write(self.template)
        self.file.close()

    def create(self, γ, b, P, R, Obs, desc="No description"):
        self.description(desc)
        self.discount(γ)
        self.variable()
        self.initial_belief(b)
        self.state_transition(P)
        self.observation(Obs)
        self.reward(R)
        self.write_to_file()
