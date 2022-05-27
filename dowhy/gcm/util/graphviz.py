import os
import tempfile
from typing import Optional, Dict, Tuple, Any

import graphviz
import networkx as nx
import numpy as np
from matplotlib import image, pyplot


def _plot_causal_graph_graphviz(causal_graph: nx.Graph,
                                display_causal_strengths: bool = True,
                                causal_strengths: Optional[Dict[Tuple[Any, Any], float]] = None,
                                filename: Optional[str] = None,
                                display_plot: bool = True) -> None:
    if causal_strengths is None:
        causal_strengths = {}

    max_strength = 0.0
    for (source, target, strength) in causal_graph.edges(data="CAUSAL_STRENGTH", default=None):
        if (source, target) not in causal_strengths:
            causal_strengths[(source, target)] = strength
        if causal_strengths[(source, target)] is not None:
            max_strength = max(max_strength, abs(causal_strengths[(source, target)]))

    if isinstance(causal_graph, nx.DiGraph):
        graphviz_directed_graph = graphviz.Digraph(engine='dot')
    else:
        graphviz_directed_graph = graphviz.Graph(engine='dot')

    for node in causal_graph.nodes:
        graphviz_directed_graph.node(str(node))

    for (source, target) in causal_graph.edges():
        causal_strength = causal_strengths[(source, target)]
        if causal_strength is not None:
            if np.isinf(causal_strength):
                causal_strength = 10000
                tmp_label = 'Inf'
            else:
                tmp_label = str(' %s' % str(int(causal_strength * 100) / 100))

            graphviz_directed_graph.edge(str(source), str(target),
                                         label=tmp_label if display_causal_strengths else None,
                                         penwidth=str(_calc_arrow_width(causal_strength, max_strength)))
        else:
            graphviz_directed_graph.edge(str(source), str(target))

    if filename is not None:
        filename, file_extension = os.path.splitext(filename)
        if file_extension == '':
            file_extension = '.pdf'
        graphviz_directed_graph.render(filename=filename, format=file_extension[1:], view=False, cleanup=True)

    if display_plot:
        _plot_as_pyplot_figure(graphviz_directed_graph)


def _calc_arrow_width(strength: float, max_strength: float):
    return 0.1 + 4.0 * float(abs(strength)) / float(max_strength)


def _plot_as_pyplot_figure(directed_graph: graphviz.Digraph) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        directed_graph.render(filename=tmp_dir_name + os.sep + 'Graph',
                              format='png',
                              view=False,
                              cleanup=True)
        img = image.imread(tmp_dir_name + os.sep + 'Graph.png')
        pyplot.imshow(img)
        pyplot.axis('off')
        pyplot.show()
