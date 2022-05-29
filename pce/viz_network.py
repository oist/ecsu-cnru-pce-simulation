import graphviz
from pce import file_utils

def network(num_nodes=2):
    f = graphviz.Digraph('pce_agent_network', format='pdf', directory = file_utils.SAVE_FOLDER)
    f.attr(rankdir='TB', ranksep="0.6", nodesep="0.8") 

    f.attr('node', shape='circle', penwidth="2")
    f.attr('edge', penwidth="0.8", arrowhead='vee', arrowsize="0.8", fontsize="10", labelloc='t')    

    for a in range(2):

        with f.subgraph(name=f'cluster_{a}') as c:

            c.attr(label=f'Agent #{a+1}')

            with c.subgraph(name='sensors') as s:
                s.attr(rank='same')
                s.attr('node', shape='circle', color="orange")
                s.node(f'S1_{a}', label='<S<SUB>1</SUB>>')

            with c.subgraph(name='neurons') as n:
                n.attr(rank='same')
                n.attr('node', shape='circle', color="blue")
                for i in range(1,num_nodes+1):
                    n.node(f'N{i}_{a}', label=f'<N<SUB>{i}</SUB>>')

            with c.subgraph(name='motors') as m:
                m.attr(rank='same')
                m.attr('node', shape='circle', color='red')
                m.node(f'M1_{a}', label='<M<SUB>1</SUB>>')
                m.node(f'M2_{a}', label='<M<SUB>2</SUB>>')

            for i in range(1,num_nodes+1):
                c.edge(f'S1_{a}', f'N{i}_{a}', label='')
            
            for i in range(1,num_nodes+1):
                for j in range(1,num_nodes+1):
                    if i==j:
                        start, end = 'n', 'n'
                    else:
                        start, end = '_', '_'
                    c.edge(f'N{i}_{a}:{start}', f'N{j}_{a}:{end}', label='')

            for i in range(1,num_nodes+1):
                c.edge(f'N{i}_{a}', f'M1_{a}', label='')
                c.edge(f'N{i}_{a}', f'M2_{a}', label='')
    

    f.view()

    

if __name__ == "__main__":
    network(2)