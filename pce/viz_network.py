import graphviz

def test():
    f = graphviz.Digraph('finite_state_machine', filename='fsm.gv')
    f.attr(rankdir='LR', size='8,5')

    f.attr('node', shape='doublecircle')
    f.node('LR_0')
    f.node('LR_3')
    f.node('LR_4')
    f.node('LR_8')

    f.attr('node', shape='circle')
    f.edge('LR_0', 'LR_2', label='SS(B)')
    f.edge('LR_0', 'LR_1', label='SS(S)')
    f.edge('LR_1', 'LR_3', label='S($end)')
    f.edge('LR_2', 'LR_6', label='SS(b)')
    f.edge('LR_2', 'LR_5', label='SS(a)')
    f.edge('LR_2', 'LR_4', label='S(A)')
    f.edge('LR_5', 'LR_7', label='S(b)')
    f.edge('LR_5', 'LR_5', label='S(a)')
    f.edge('LR_6', 'LR_6', label='S(b)')
    f.edge('LR_6', 'LR_5', label='S(a)')
    f.edge('LR_7', 'LR_8', label='S(b)')
    f.edge('LR_7', 'LR_5', label='S(a)')
    f.edge('LR_8', 'LR_6', label='S(b)')
    f.edge('LR_8', 'LR_5', label='S(a)')

    f.view()

def network(num_nodes=2):
    f = graphviz.Digraph('pce_agent_network', filename='pce_agent.gv')
    f.attr(rankdir='TB', ranksep="0.6", nodesep="0.6") 

    f.attr('node', shape='circle', penwidth="2")
    f.attr('edge', penwidth="0.8", arrowhead='vee', arrowsize="0.8", fontsize="10", labelloc='t')    
    
    with f.subgraph(name='sensors') as s:
        s.attr(rank='same')
        s.attr('node', shape='circle', color="orange")
        s.node('S1')

    with f.subgraph(name='neurons') as n:
        n.attr(rank='same')
        n.attr('node', shape='circle', color="blue")
        for i in range(1,num_nodes+1):
            n.node(f'N{i}', label=f'<N<SUB>{i}</SUB>>')            

    with f.subgraph(name='motors') as m:
        m.attr(rank='same')
        m.attr('node', shape='circle', color='red')
        m.node('M1', label='<M<SUB>1</SUB>>')
        m.node('M2', label='<M<SUB>2</SUB>>')

    for i in range(1,num_nodes+1):
        f.edge('S1', f'N{i}', label='')
    
    for i in range(1,num_nodes+1):
        for j in range(1,num_nodes+1):
            if i==j:
                if i==1:
                    start, end = 'nw', 'sw'
                elif i==num_nodes:
                    start, end = 'ne', 'se'
                else:
                    start, end = 'ne', 'se'
            else:
                start, end = '_', '_'
            f.edge(f'N{i}:{start}', f'N{j}:{end}', label='')

    for i in range(1,num_nodes+1):
        f.edge(f'N{i}', 'M1', label='')
        f.edge(f'N{i}', 'M2', label='')
    

    f.view()

    

if __name__ == "__main__":
    # test()
    network(2)