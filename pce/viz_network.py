import graphviz
from pce import file_utils

def get_phenotype_value(phenotypes, a, key, idx):
    if phenotypes is None:
        return ''
    array = phenotypes[a][key]
    value = array[idx]
    value = f'{value:.2f}' # to string 2 decimal
    return value

def plot_network(num_nodes=2, phenotypes=None):
    f = graphviz.Digraph('pce_agent_network', format='pdf', directory = file_utils.SAVE_FOLDER)
    f.attr(rankdir='TB', ranksep="0.6", nodesep="0.8") 

    f.attr('node', shape='circle', penwidth="2")
    f.attr('edge', penwidth="0.8", arrowhead='vee', arrowsize="0.8", fontsize="10") # , labelloc='t'

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
            
            # TODO: neural_taus
            # TODO: neural_biases
            # TODO: neural_gains

            with c.subgraph(name='motors') as m:
                m.attr(rank='same')
                m.attr('node', shape='circle', color='red')
                m.node(f'M1_{a}', label='<M<SUB>1</SUB>>')
                m.node(f'M2_{a}', label='<M<SUB>2</SUB>>')

            for i in range(num_nodes):
                label = get_phenotype_value(phenotypes, a, 'sensor_weights', (0,i))
                c.edge(f'S1_{a}', f'N{i+1}_{a}', taillabel=label)
                # TODO: sensor_gains
                # TODO: sensor_biases
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i==j:
                        start, end = 'n', 'n'
                    else:
                        start, end = '_', '_'
                    label = get_phenotype_value(phenotypes, a, 'neural_weights', (i,j))
                    c.edge(f'N{i+1}_{a}:{start}', f'N{j+1}_{a}:{end}', label=label)

            for i in range(num_nodes):
                for j in range(2):
                    # motor_weights
                    label = get_phenotype_value(phenotypes, a, 'motor_weights', (i,j))
                    c.edge(f'N{i+1}_{a}:s', f'M{j+1}_{a}', taillabel=label)
                    # TODO: motor_gains
                    # TODO: motor_biases
    

    f.view()

    

if __name__ == "__main__":
    plot_network(2)