from .base import SearchResult
from .bfs import bfs_search
from .dfs import dfs_search
from .ucs import ucs_search
from .ids import ids_search
from .greedy import greedy_search
from .astar import astar_search
from .weighted_astar import weighted_astar_search

ALL_ALGORITHMS = {
    "BFS": bfs_search,
    "DFS": dfs_search,
    "UCS": ucs_search,
    "IDS": ids_search,
    "Greedy": greedy_search,
    "A*": astar_search,
    "Weighted A*": weighted_astar_search,
}
