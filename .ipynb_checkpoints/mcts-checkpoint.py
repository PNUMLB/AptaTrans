import numpy as np
import timeit
import torch
from utils import rna2vec

#Node
class Node:
    #init
    def __init__(self, letter="", parent=None, root=False, last=False, depth=0, states=8):
        self.exploitation_score = 0 # Exploitaion score
        self.visits = 1 #How many visits
        self.letter = letter #This node's letter
        self.parent = parent #This node's parent node
        self.states = states #How many states in node
        self.children = np.array([None for _ in range(self.states)]) #This node's children
        self.children_stat = np.zeros(self.states, dtype=bool) #Which stat are expanded
        self.root = root # Is root? boolean
        self.last = last # Is last node?
        self.depth = depth # My depth
        self.letters =["A_", "C_", "G_", "U_", "_A", "_C", "_G", "_U"]

    
    #next_node
    def next_node(self, child=0): #Return next node
        assert self.children_stat[child] == True, "No child in here."
        
        return self.children[child]
    
    #back_parent
    def back_parent(self): #Go back to parent
        return self.parent, letters_map[self.letter]
    
    #generate_child
    def generate_child(self, child=0, last=False): #Generate child
        assert self.children_stat[child] == False, "Already tree generated child at here"
        
        self.children[child] = Node(letter=self.letters[child], parent=self, last=last, depth=self.depth+1, states=self.states) #New node
        self.children_stat[child] = True #Stat = True
        
        return self.children[child]
    
    #backpropagation
    def backpropagation(self, score=0):
        self.visits += 1 # +1 to visit
        if self.root == True: # if root, then stop
            return self.exploitation_score
        
        else:
            self.exploitation_score += score #Add score to exploitation score
            return self.parent.backpropagation(score=score) #Backpropagation to parent node
    
    #UCT
    def UCT(self):
        return (self.exploitation_score / self.visits) + np.sqrt(np.log(self.parent.visits) / (2 * self.visits)) #UCT score
    

#MCTS
class MCTS:
    def __init__(self, target_encoded, depth=20, iteration=1000, states=8, target_protein="", device='cpu'):
        self.states = states #How many states
        self.root = Node(letter="", parent=None, root=True, last=False, states=self.states) #root node
        self.depth = depth #Maximum depth
        self.iteration = iteration #iteration for expand 
        self.target_protein = target_protein #target protein's amino acid sequence
        self.device = device
        self.encoded_targetprotein = target_encoded
        self.base = ""
        self.candidate = ""
        self.letters =["A_", "C_", "G_", "U_", "_A", "_C", "_G", "_U"]


    def make_candidate(self, classifier):
        now = self.root
        n = 0 # rounds
        start_time = timeit.default_timer() #timer start
        
        while len(self.base) < self.depth * 2: #If now is last node, then stop
            n += 1
            print(n, "round start!!!")
            for _ in range(self.iteration):
                now = self.select(classifier, now=now) #Select & Expand

            terminate_time = timeit.default_timer()
            time = terminate_time-start_time
            
            base = self.find_best_subsequence() #Find best subsequence
            self.base = base

            print("best subsequence:", base)
            print("Depth:", int(len(base)/2))
            print("%02d:%02d:%2f" % ((time//3600), (time//60)%60, time%60))
            print("=" * 80)

            self.root = Node(letter="", parent=None, root=True, last=False, states=self.states, depth=len(self.base)/2)
            now = self.root
            
        self.candidate = self.base
        
        return self.candidate
            
    #selection
    def select(self, classifier, now=None):
        if now.depth == self.depth: #If last node, then stop
            return self.root
        
        next_node = 0
        if np.sum(now.children_stat) == self.states: #If every child is expanded, then go to best child
            best = 0
            for i in range(self.states):
                if best < now.children[i].UCT():
                    next_node = i
                    best = now.children[i].UCT()
                    
        else: #If not, then random
            next_node = np.random.randint(0, self.states)
            if now.children_stat[next_node] == False: #If selected child is not expanded, then expand and simulate
                next_node = self.expand(classifier, child=next_node, now=now)
    
                return self.root #start iteration at this node
            
        return now.next_node(child=next_node)
    
    #expand
    def expand(self, classifier, child=None, now=None):
        last = False
        if now.depth == (self.depth-1): #If depth of this node is maximum depth -1, then next node is last
            last = True
        
        expanded_node = now.generate_child(child=child, last=last) #Expand
        
        score = self.simulate(classifier, target=expanded_node)  #Simulate
        expanded_node.backpropagation(score=score) #Backporpagation
        
        return child
    
    #simulate
    def simulate(self, classifier, target=None):
        now = target #Target node
        sim_seq = ""
        
        while now.root != True: #Parent's letters
            sim_seq = now.letter + sim_seq
            now = now.parent
            
        sim_seq = self.base + sim_seq
        
        for i in range((self.depth * 2) - len(sim_seq)): #Random child letters
            r = np.random.randint(0,self.states)
            sim_seq += self.letters[r]
        
        sim_seq = self.reconstruct(sim_seq)
        scores = []

        classifier.eval()
        with torch.no_grad():
            sim_seq = self.reconstruct(sim_seq)
            sim_seq = np.array([sim_seq])
            apta = torch.tensor(rna2vec(sim_seq), dtype=torch.int64).to(self.device)

            
            score = classifier(apta, self.encoded_targetprotein)

        return score
    
    #recommend
    def get_candidate(self):
        return self.reconstruct(self.candidate)
    
    def find_best_subsequence(self):
        now = self.root
        stop = False
        base = self.base
        
        for _ in range((self.depth*2) - len(base)):
            best = 0
            next_node = 0
            for j in range(self.states):
                if now.children_stat[j] == True:
                    if best < now.children[j].UCT():
                        next_node = j
                        best = now.children[j].UCT()

            now = now.next_node(child=next_node)
            base += now.letter
            
            if np.sum(now.children_stat) == 0:
                break
                
        return base
            
    #reconstruct
    def reconstruct(self, seq=""):
        r_seq = ""
        for i in range(0, len(seq), 2):
            if seq[i] == '_':
                r_seq = r_seq + seq[i+1]
            else:
                r_seq = seq[i] + r_seq
        return r_seq
    
    def reset(self):
        self.base = ""
        self.candidate = ""
        self.root = Node(letter="", parent=None, root=True, last=False, states=self.states)