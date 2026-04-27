"""
Neural Network Visualization Module

Draws small directed weighted neural networks with modulatory connections.
Uses the "dummy node trick" to visualize modulatory connections that target
regular connections rather than neurons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Wedge
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import yaml
from pathlib import Path


# ===== Configuration Loading =====

def load_network_viz_config(config_name: Union[str, int]) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, str]]:
    """
    Load network visualization configuration from YAML file.
    
    Args:
        config_name: Config identifier (e.g., '11' -> loads 'network_viz_11.yaml')
    
    Returns:
        (neuron_positions, neuron_types) tuple ready for draw_network()
    
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config file is malformed
    """
    config_name = str(config_name)
    
    # Find workspace root (contains 'data', 'simulate', 'configs' folders)
    # analysis_tools is now at root level, so navigate from there
    module_dir = Path(__file__).resolve().parent  # network_visualization folder
    current_path = module_dir
    workspace_root = None
    
    while current_path.parent != current_path:  # While not at filesystem root
        if (current_path / "data").exists() and (current_path / "simulate").exists() and (current_path / "configs").exists():
            workspace_root = current_path
            break
        current_path = current_path.parent
    
    if workspace_root is None:
        raise RuntimeError("Could not find workspace root. Searched parent directories for 'data', 'simulate', and 'configs' folders.")
    
    config_path = workspace_root / "configs" / f"network_viz_{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Network visualization config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if not config:
        raise ValueError(f"Config file is empty: {config_path}")
    
    # Parse neuron positions
    positions_raw = config.get('neuron_positions', {})
    neuron_positions = {}
    for nid, pos in positions_raw.items():
        nid_int = int(nid)
        if isinstance(pos, list) and len(pos) == 2:
            neuron_positions[nid_int] = tuple(pos)
        else:
            neuron_positions[nid_int] = pos
    
    # Parse neuron types
    neuron_types = {}
    types_raw = config.get('neuron_types', {})
    for nid, ntype in types_raw.items():
        neuron_types[int(nid)] = ntype
    
    return neuron_positions, neuron_types


# ===== Automatic Layout Computation =====

def compute_layout(connections: Union[pd.DataFrame, List[Dict]],
                   neuron_types: Optional[Dict[int, str]] = None,
                   use_graphviz: bool = True,
                   spring_k: float = 2.0,
                   spring_iterations: int = 50) -> Dict[int, Tuple[float, float]]:
    """
    Compute neuron positions automatically from network structure.
    
    Uses graphviz's dot layout (if available) for hierarchical directed graphs,
    or falls back to spring layout for a more general force-directed layout.
    
    Args:
        connections: Regular connections as DataFrame or list of dicts with 'src' and 'tgt'
        
        neuron_types: Optional dict mapping neuron_id -> type for improved layout hints.
                     Not currently used, but kept for future enhancements.
        
        use_graphviz: If True, try graphviz first; if False or unavailable, use spring layout
        
        spring_k: Repulsive force constant for spring layout (larger = more spread out)
        
        spring_iterations: Number of iterations for spring layout algorithm
    
    Returns:
        Dict mapping neuron_id -> (x, y) with positions normalized to [0, 1] range
    
    Raises:
        ValueError: If connections is empty
    """
    # Convert to DataFrame if needed
    if isinstance(connections, list):
        connections = pd.DataFrame(connections)
    
    if len(connections) == 0:
        raise ValueError("connections cannot be empty")
    
    # Build directed graph
    G = nx.DiGraph()
    for _, row in connections.iterrows():
        src = int(row['src'])
        tgt = int(row['tgt'])
        G.add_edge(src, tgt)
    
    # Try graphviz first if requested
    if use_graphviz:
        try:
            pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
            print("Using graphviz (dot) layout")
        except Exception as e:
            print(f"Graphviz layout failed ({type(e).__name__}), falling back to spring layout")
            pos = nx.spring_layout(G, k=spring_k, iterations=spring_iterations, seed=42)
    else:
        print(f"Using spring layout with k={spring_k}")
        pos = nx.spring_layout(G, k=spring_k, iterations=spring_iterations, seed=42)
    
    # Normalize positions to [0, 1] range
    if len(pos) == 0:
        raise ValueError("Failed to compute layout - no nodes in graph")
    
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    
    # Handle edge case where all nodes are at same position
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    
    pos_normalized = {}
    for nid, (x, y) in pos.items():
        normalized_x = (x - x_min) / x_range if x_range > 0 else 0.5
        normalized_y = (y - y_min) / y_range if y_range > 0 else 0.5
        pos_normalized[nid] = (normalized_x, normalized_y)
    
    return pos_normalized


# ===== Configuration Constants =====

COLORS = {
    'neuron_fill': '#D3AF37',           # Gold/Brass
    'input_edge': '#8B6F47',            # Light brown
    'output_edge': '#654321',           # Darker brown
    'hidden_edge': '#D3AF37',           # Same as fill (no visible edge)
    'excitatory': '#0B3D2E',            # Dark green
    'inhibitory': '#721817',            # Dark red/brown
    'modulatory_potentiation': '#0B3D2E',  # Dark green (positive modulation, dashed)
    'modulatory_depression': '#721817',    # Dark red/brown (negative modulation, dashed)
}

SIZES = {
    'neuron': 800,
    'dummy': 80,
    'label_fontsize': 14,
    'weight_fontsize': 12,
}

STYLES = {
    'regular_connection': '-',    # Solid line
    'modulatory_connection': '--', # Dashed line
    'dummy_marker': 'D',          # Diamond
}


# ===== Helper Functions =====

def _compute_dummy_position(src_pos: Tuple[float, float], 
                           tgt_pos: Tuple[float, float],
                           ratio: float = 0.5) -> Tuple[float, float]:
    """
    Compute position of dummy node along a connection.
    
    Args:
        src_pos: (x, y) position of source neuron
        tgt_pos: (x, y) position of target neuron
        ratio: Position along the edge (0.5 = midpoint)
    
    Returns:
        (x, y) position for dummy node
    """
    x = src_pos[0] + ratio * (tgt_pos[0] - src_pos[0])
    y = src_pos[1] + ratio * (tgt_pos[1] - src_pos[1])
    return (x, y)


def _get_dummy_id(src: int, tgt: int) -> str:
    """Generate deterministic ID for dummy node."""
    return f"dummy_{src}_{tgt}"


def _shorten_edge_positions(src_pos: Tuple[float, float],
                           tgt_pos: Tuple[float, float],
                           offset: float = 0.07) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Shorten edge positions by moving start/end points away from neuron centers.
    
    This exposes arrowheads and prevents them from being hidden behind neuron circles.
    
    Args:
        src_pos: Source neuron position
        tgt_pos: Target neuron position
        offset: Distance to offset from neuron centers (in data units)
    
    Returns:
        (shortened_src_pos, shortened_tgt_pos) tuple
    """
    # Calculate direction vector
    dx = tgt_pos[0] - src_pos[0]
    dy = tgt_pos[1] - src_pos[1]
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist < 1e-6:  # Avoid division by zero
        return src_pos, tgt_pos
    
    # Normalize
    dx_norm = dx / dist
    dy_norm = dy / dist
    
    # Shorten from both ends
    new_src = (src_pos[0] + offset * dx_norm, src_pos[1] + offset * dy_norm)
    new_tgt = (tgt_pos[0] - offset * dx_norm, tgt_pos[1] - offset * dy_norm)
    
    return new_src, new_tgt


def _shorten_edge_source_only(src_pos: Tuple[float, float],
                             tgt_pos: Tuple[float, float],
                             offset: float = 0.07) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Shorten edge position at the source only (keeps target unchanged).
    
    Used for modulatory arrows that should reach exactly to the dummy node.
    
    Args:
        src_pos: Source neuron position
        tgt_pos: Target position (usually dummy node)
        offset: Distance to offset from source neuron center
    
    Returns:
        (shortened_src_pos, unchanged_tgt_pos) tuple
    """
    # Calculate direction vector
    dx = tgt_pos[0] - src_pos[0]
    dy = tgt_pos[1] - src_pos[1]
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist < 1e-6:  # Avoid division by zero
        return src_pos, tgt_pos
    
    # Normalize
    dx_norm = dx / dist
    dy_norm = dy / dist
    
    # Shorten only the source end
    new_src = (src_pos[0] + offset * dx_norm, src_pos[1] + offset * dy_norm)
    
    return new_src, tgt_pos


def _get_curved_arrow_position(src_pos: Tuple[float, float],
                               tgt_pos: Tuple[float, float],
                               curve_direction: int = 1,
                               rad: float = 0.2,
                               t: float = 0.5) -> Tuple[float, float]:
    """
    Calculate a point on a curved arrow path (arc3 style).
    
    Used to position text annotations on the actual curve rather than a straight line.
    
    Args:
        src_pos: Source position
        tgt_pos: Target position
        curve_direction: 1 for right curve, -1 for left curve
        rad: Curvature radius (same as arc3 rad parameter)
        t: Position along the curve (0.5 = midpoint)
    
    Returns:
        (x, y) position on the curved path
    """
    # Calculate the straight line midpoint
    mid_x = src_pos[0] + t * (tgt_pos[0] - src_pos[0])
    mid_y = src_pos[1] + t * (tgt_pos[1] - src_pos[1])
    
    # Calculate direction vector
    dx = tgt_pos[0] - src_pos[0]
    dy = tgt_pos[1] - src_pos[1]
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist < 1e-6:  # No curvature if points are too close
        return (mid_x, mid_y)
    
    # Calculate perpendicular vector (rotated 90 degrees)
    # For right curve (clockwise): use clockwise perpendicular
    perp_x = dy / dist
    perp_y = -dx / dist
    
    # For arc3, the control point is offset perpendicular to the center line by:
    # control_point = midpoint + rad * distance * perpendicular_unit_vector
    # For a quadratic Bezier B(t) = (1-t)² * P0 + 2(1-t)t * P1 + t² * P2
    # At t=0.5 (midpoint of curve):
    # B(0.5) = 0.25*P0 + 0.5*P1 + 0.25*P2
    #        = midpoint + 0.5*rad*distance*perpendicular_unit_vector
    offset = curve_direction * rad * dist * 0.5
    
    # Apply perpendicular offset
    curved_x = mid_x + offset * perp_x
    curved_y = mid_y + offset * perp_y
    
    return (curved_x, curved_y)


def _draw_self_connection(ax, pos: Tuple[float, float], weight: float, 
                         neuron_size: float = 0.03):
    """
    Draw a self-connection as a circular loop at the neuron position.
    
    Args:
        ax: Matplotlib axis
        pos: (x, y) position of neuron
        weight: Connection weight
        neuron_size: Radius for the loop
    """
    color = COLORS['excitatory'] if weight >= 0 else COLORS['inhibitory']
    linewidth = 1 + 6 * abs(weight)  # Scale with weight magnitude
    
    # Draw loop as a circle centered at the neuron position
    circle = Circle(pos, 
                   radius=neuron_size * 1.8,
                   fill=False,
                   edgecolor=color,
                   linewidth=linewidth,
                   zorder=2)
    ax.add_patch(circle)
    
    # Add small arrowhead at the top of the circle
    arrow_angle = np.pi / 2  # Top of circle
    arrow_x = pos[0] + neuron_size * 1.6 * np.cos(arrow_angle + 0.3)
    arrow_y = pos[1] + neuron_size * 1.6 * np.sin(arrow_angle + 0.3)
    
    ax.annotate('', xy=(arrow_x, arrow_y),
               xytext=(pos[0], pos[1] + neuron_size * 1.6),
               arrowprops=dict(arrowstyle='->', color=color, lw=linewidth))


def _draw_curved_arrow(ax, src_pos: Tuple[float, float], 
                      tgt_pos: Tuple[float, float],
                      weight: float, 
                      curve_direction: int = 1,
                      style: str = '-',
                      color: Optional[str] = None,
                      alpha: float = 0.7,
                      skip_shorten: bool = False):
    """
    Draw a curved arrow between two positions.
    
    Args:
        ax: Matplotlib axis
        src_pos: (x, y) source position
        tgt_pos: (x, y) target position
        weight: Connection weight (affects thickness)
        curve_direction: 1 for right curve, -1 for left curve
        style: Line style ('-' for solid, '--' for dashed)
        color: Line color (auto-determined from weight if None)
        alpha: Transparency
        skip_shorten: If True, don't shorten positions (already shortened)
    """
    if color is None:
        color = COLORS['excitatory'] if weight >= 0 else COLORS['inhibitory']
    
    # Shorten positions to expose arrowheads (unless already shortened)
    if not skip_shorten:
        src_pos, tgt_pos = _shorten_edge_positions(src_pos, tgt_pos, offset=0.07)
    
    linewidth = 0.5 + 7.5 * abs(weight)
    rad = 0.2 * curve_direction  # Curvature
    
    arrow = FancyArrowPatch(
        src_pos, tgt_pos,
        arrowstyle='-|>',
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=15,
        linewidth=linewidth,
        color=color,
        linestyle=style,
        alpha=alpha,
        zorder=5
    )
    ax.add_patch(arrow)


def _draw_straight_arrow(ax, src_pos: Tuple[float, float],
                        tgt_pos: Tuple[float, float],
                        weight: float,
                        style: str = '-',
                        color: Optional[str] = None,
                        alpha: float = 0.7,
                        skip_shorten: bool = False):
    """Draw a straight arrow (used when no bidirectional conflict)."""
    if color is None:
        color = COLORS['excitatory'] if weight >= 0 else COLORS['inhibitory']
    
    # Shorten positions to expose arrowheads (unless already shortened)
    if not skip_shorten:
        src_pos, tgt_pos = _shorten_edge_positions(src_pos, tgt_pos, offset=0.07)
    
    linewidth = 0.5 + 7.5 * abs(weight)
    
    arrow = FancyArrowPatch(
        src_pos, tgt_pos,
        arrowstyle='-|>',
        mutation_scale=15,
        linewidth=linewidth,
        color=color,
        linestyle=style,
        alpha=alpha,
        zorder=5
    )
    ax.add_patch(arrow)


def _detect_bidirectional_edges(connections: pd.DataFrame) -> set:
    """
    Detect pairs of neurons with bidirectional connections.
    
    Returns:
        Set of frozensets containing neuron pairs with bidirectional edges
    """
    bidirectional = set()
    edge_set = set()
    
    for _, row in connections.iterrows():
        src, tgt = int(row['src']), int(row['tgt'])
        if src == tgt:  # Skip self-connections
            continue
        
        reverse_edge = (tgt, src)
        if reverse_edge in edge_set:
            bidirectional.add(frozenset([src, tgt]))
        edge_set.add((src, tgt))
    
    return bidirectional


def _get_neuron_color(neuron_id: int, neuron_types: Dict[int, str]) -> tuple:
    """
    Get fill and edge colors for a neuron based on its type.
    
    Args:
        neuron_id: Neuron identifier
        neuron_types: Dict mapping neuron_id -> type ('input', 'output', 'hidden')
    
    Returns:
        (fill_color, edge_color) tuple
    """
    neuron_type = neuron_types.get(neuron_id, 'hidden')
    fill = COLORS['neuron_fill']
    
    if neuron_type == 'input':
        edge = COLORS['input_edge']
    elif neuron_type == 'output':
        edge = COLORS['output_edge']
    else:  # hidden
        edge = COLORS['hidden_edge']
    
    return fill, edge


# ===== Main Drawing Function =====

def draw_network(neurons: Dict[int, Dict],
                connections: Union[pd.DataFrame, List[Dict]],
                modulatory: Optional[Union[pd.DataFrame, List[Dict]]] = None,
                ax: Optional[plt.Axes] = None,
                title: str = '',
                show_weights: bool = False,
                weight_column: str = 'weight') -> plt.Axes:
    """
    Draw a neural network with regular and modulatory connections.
    
    Args:
        neurons: Dict mapping neuron_id -> {'pos': (x, y), 'type': 'input'|'output'|'input_output'|'hidden'}
                 Example: {0: {'pos': (0, 0), 'type': 'input'},
                          1: {'pos': (1, 0), 'type': 'hidden'}}
        
        connections: Regular connections as DataFrame or list of dicts with columns:
                    - 'src': source neuron ID
                    - 'tgt': target neuron ID
                    - weight_column: connection weight (default 'weight')
        
        modulatory: Modulatory connections as DataFrame or list of dicts with columns:
                   - 'target_src': source of the target connection
                   - 'target_tgt': target of the target connection
                   - 'modulator_src': neuron sending the modulatory signal
                   - 'modulation_weight': strength of modulation
        
        ax: Matplotlib axis (creates new if None)
        title: Plot title
        show_weights: Whether to display weight values on connections
        weight_column: Name of the weight column in connections DataFrame
    
    Returns:
        The matplotlib axis with the drawn network
    """
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Convert to DataFrame if needed
    if isinstance(connections, list):
        connections = pd.DataFrame(connections)
    if modulatory is not None and isinstance(modulatory, list):
        modulatory = pd.DataFrame(modulatory)
    
    # Ensure weight column exists
    if weight_column not in connections.columns:
        connections[weight_column] = 1.0
    
    # Extract neuron types for coloring
    neuron_types = {nid: info.get('type', 'hidden') for nid, info in neurons.items()}
    positions = {nid: info['pos'] for nid, info in neurons.items()}
    
    # Detect bidirectional edges for curved drawing
    bidirectional_pairs = _detect_bidirectional_edges(connections)
    
    # ===== Draw Regular Connections =====
    drawn_edges = set()
    dummy_positions = {}  # Store dummy node positions for modulatory connections
    
    for _, row in connections.iterrows():
        src, tgt = int(row['src']), int(row['tgt'])
        weight = float(row[weight_column])
        
        src_pos = positions[src]
        tgt_pos = positions[tgt]
        
        # Handle self-connections specially
        if src == tgt:
            _draw_self_connection(ax, src_pos, weight)
            # Dummy node for self-connection (offset above)
            dummy_positions[_get_dummy_id(src, tgt)] = (
                src_pos[0], src_pos[1] + 0.08
            )
            continue
        
        # Check if this edge is part of a bidirectional pair
        edge_pair = frozenset([src, tgt])
        is_bidirectional = edge_pair in bidirectional_pairs
        
        # Shorten positions once for both drawing and dummy node calculation
        shortened_src, shortened_tgt = _shorten_edge_positions(src_pos, tgt_pos, offset=0.07)
        
        if is_bidirectional:
            # Draw curved to avoid overlap
            # Always curve right from the source perspective.
            # When A→B curves right and B→A also curves right from B's perspective,
            # they automatically curve away from each other (like traffic on the right side of the road)
            _draw_curved_arrow(ax, shortened_src, shortened_tgt, weight, 
                             curve_direction=1, skip_shorten=True)
        else:
            # Draw straight arrow
            _draw_straight_arrow(ax, shortened_src, shortened_tgt, weight, skip_shorten=True)
        
        drawn_edges.add((src, tgt))
        
        # Compute and store dummy node position based on shortened positions
        # (where the arrows actually are, not where the neurons are)
        dummy_id = _get_dummy_id(src, tgt)
        dummy_positions[dummy_id] = _compute_dummy_position(shortened_src, shortened_tgt)
        
        # Always show weight value for regular connections
        mid_x, mid_y = dummy_positions[dummy_id]
        ax.text(mid_x, mid_y, f'{weight:.2f}',
               fontsize=SIZES['weight_fontsize'],
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
               zorder=10)
    
    # ===== Draw Neurons =====
    for neuron_id, info in neurons.items():
        pos = info['pos']
        fill_color, edge_color = _get_neuron_color(neuron_id, neuron_types)
        
        ax.scatter(pos[0], pos[1],
                  s=SIZES['neuron'],
                  c=fill_color,
                  edgecolors=edge_color,
                  linewidths=2,
                  zorder=4,
                  alpha=0.9)
        
        # Label neuron
        ax.text(pos[0], pos[1], str(neuron_id),
               fontsize=SIZES['label_fontsize'],
               ha='center', va='center',
               fontweight='bold',
               color='black',
               zorder=5)
    
    # ===== Finalize Plot =====
    ax.set_aspect('equal')
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['neuron_fill'], edgecolor=COLORS['input_edge'], 
                      linewidth=2, label='Input Neuron'),
        mpatches.Patch(facecolor=COLORS['neuron_fill'], edgecolor=COLORS['output_edge'], 
                      linewidth=2, label='Output Neuron'),
        mpatches.Patch(facecolor=COLORS['neuron_fill'], edgecolor=COLORS['hidden_edge'], 
                      linewidth=2, label='Hidden Neuron'),
        plt.Line2D([0], [0], color=COLORS['excitatory'], linewidth=2, 
                   linestyle='-', label='Excitatory Connection'),
        plt.Line2D([0], [0], color=COLORS['inhibitory'], linewidth=2,
                   linestyle='-', label='Inhibitory Connection'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(1.02, 1), fontsize=11)
    
    # Adjust limits to fit all elements
    all_x = [pos[0] for pos in positions.values()]
    all_y = [pos[1] for pos in positions.values()]
    margin = 0.15
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    return ax


def draw_network_over_time(wiring_csv: str,
                           timesteps: List[Union[str, int]],
                           modulation_csv: Optional[str] = None,
                           neuron_positions: Optional[Dict[int, Tuple[float, float]]] = None,
                           neuron_types: Optional[Dict[int, str]] = None,
                           title: str = '') -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Draw neural networks at multiple timesteps with a shared legend.
    
    Creates a figure with subplots (facets) showing network evolution over time.
    A single legend is displayed at the end covering all facets.
    
    Args:
        wiring_csv: Path to wiring CSV file with columns 'src', 'tgt', and weight columns for each timestep
                   Supported timestep formats:
                   - 'initial': uses 'weight_initial' column
                   - int or str int (e.g., 50): looks for 'weight_t50' or 'weight_timestep_50' columns
                   - str like 'final_run_0001': uses 'weight_final_run_0001' column
        
        timesteps: List of timesteps to plot (e.g., ['initial', 50, 100] or ['initial', 'final_run_0001'])
        
        modulation_csv: Path to modulation CSV (same modulation used for all timesteps)
        
        neuron_positions: Manual positions for neurons. If None, auto-generates a circular layout
        
        neuron_types: Dict mapping neuron_id -> type. If None, infers from data
        
        title: Main figure title
    
    Returns:
        (fig, axes) tuple where fig is the matplotlib figure and axes is a list of axes (one per timestep)
    """
    # Load the full wiring data
    wiring_data = pd.read_csv(wiring_csv)
    
    # Load modulatory if provided
    modulatory = None
    if modulation_csv and Path(modulation_csv).exists():
        modulatory = pd.read_csv(modulation_csv)
    
    # Get all unique neuron IDs
    all_neurons = set(wiring_data['src'].unique()) | set(wiring_data['tgt'].unique())
    
    # Auto-generate positions if not provided
    if neuron_positions is None:
        n = len(all_neurons)
        neuron_positions = {}
        for i, nid in enumerate(sorted(all_neurons)):
            angle = 2 * np.pi * i / n
            neuron_positions[nid] = (np.cos(angle), np.sin(angle))
    
    # Auto-infer neuron types if not provided
    if neuron_types is None:
        neuron_types = {}
        sources = set(wiring_data['src'].unique())
        targets = set(wiring_data['tgt'].unique())
        
        for nid in all_neurons:
            if nid in sources and nid in targets:
                neuron_types[nid] = 'hidden'
            elif nid in sources:
                neuron_types[nid] = 'input'
            elif nid in targets:
                neuron_types[nid] = 'output'
            else:
                neuron_types[nid] = 'hidden'
    
    # Build neurons dict
    neurons = {}
    for nid in all_neurons:
        neurons[nid] = {
            'pos': neuron_positions[nid],
            'type': neuron_types.get(nid, 'hidden')
        }
    
    # Map timesteps to weight columns
    weight_columns = {}
    for ts in timesteps:
        if ts == 'initial':
            weight_col = 'weight_initial'
        elif isinstance(ts, (int, np.integer)):
            # Try different column naming conventions
            ts_str = str(ts).zfill(4)  # Padding like 0050
            candidates = [
                f'weight_t{ts}',
                f'weight_t{ts_str}',
                f'weight_timestep_{ts}',
                f'weight_timestep_{ts_str}',
            ]
            weight_col = None
            for candidate in candidates:
                if candidate in wiring_data.columns:
                    weight_col = candidate
                    break
            if weight_col is None:
                raise ValueError(f"Could not find weight column for timestep {ts} in wiring CSV. "
                               f"Available columns: {list(wiring_data.columns)[:10]}...")
        else:
            # Assume it's a full column name or final_run_XXXX format
            ts_str = str(ts)
            if ts_str.startswith('final_run_'):
                weight_col = f'weight_{ts_str}'
            else:
                weight_col = ts_str
            
            if weight_col not in wiring_data.columns:
                raise ValueError(f"Column {weight_col} not found in wiring CSV")
        
        weight_columns[ts] = weight_col
    
    # Create figure with subplots (one per timestep)
    n_timesteps = len(timesteps)
    fig, axes = plt.subplots(1, n_timesteps, figsize=(5*n_timesteps, 5))
    
    # Ensure axes is always a list (even if only 1 subplot)
    if n_timesteps == 1:
        axes = [axes]
    
    # Draw network at each timestep
    axes_list = []
    for ax, ts in zip(axes, timesteps):
        weight_col = weight_columns[ts]
        
        # Create connections dataframe with the appropriate weight column
        connections = wiring_data[['src', 'tgt', weight_col]].copy()
        connections = connections.rename(columns={weight_col: 'weight'})
        
        # Draw the network
        draw_network(
            neurons=neurons,
            connections=connections,
            modulatory=modulatory,
            ax=ax,
            title=f'Timestep: {ts}',
            show_weights=False,
            weight_column='weight'
        )
        
        # Remove the individual legend from this subplot
        if ax.get_legend():
            ax.get_legend().remove()
        
        axes_list.append(ax)
    
    # Add a single shared legend at the end
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['neuron_fill'], edgecolor=COLORS['input_edge'], 
                      linewidth=2, label='Input Neuron'),
        mpatches.Patch(facecolor=COLORS['neuron_fill'], edgecolor=COLORS['output_edge'], 
                      linewidth=2, label='Output Neuron'),
        mpatches.Patch(facecolor=COLORS['neuron_fill'], edgecolor=COLORS['hidden_edge'], 
                      linewidth=2, label='Hidden Neuron'),
        plt.Line2D([0], [0], color=COLORS['excitatory'], linewidth=2, 
                   linestyle='-', label='Excitatory Connection'),
        plt.Line2D([0], [0], color=COLORS['inhibitory'], linewidth=2,
                   linestyle='-', label='Inhibitory Connection'),
    ]
    
    # Add the shared legend outside the subplots
    fig.legend(handles=legend_elements, loc='center right', 
              bbox_to_anchor=(0.98, 0.5), fontsize=11)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    
    return fig, axes_list


def draw_network_panels(wiring_csv: str,
                        weight_columns: List[str],
                        panel_labels: List[str],
                        modulation_csv: Optional[str] = None,
                        neuron_positions: Optional[Dict[int, Tuple[float, float]]] = None,
                        neuron_types: Optional[Dict[int, str]] = None,
                        title: str = '') -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Draw neural networks in multiple panels (facets) with a shared legend.
    
    Creates a figure with subplots showing networks from different weight columns.
    A single legend is displayed outside all panels.
    
    Args:
        wiring_csv: Path to wiring CSV file with columns 'src', 'tgt', and weight columns
        
        weight_columns: List of weight column names to plot (e.g., ['weight_initial', 'weight_final_run_0001', ...])
        
        panel_labels: List of labels for each panel (e.g., ['Initial', 'Run 1', 'Run 2', 'Run 3'])
        
        modulation_csv: Path to modulation CSV (same modulation used for all panels)
        
        neuron_positions: Manual positions for neurons. If None, auto-generates a circular layout
        
        neuron_types: Dict mapping neuron_id -> type. If None, infers from data
        
        title: Main figure title
    
    Returns:
        (fig, axes) tuple where fig is the matplotlib figure and axes is a list of axes (one per panel)
    """
    if len(weight_columns) != len(panel_labels):
        raise ValueError(f"weight_columns and panel_labels must have same length. "
                        f"Got {len(weight_columns)} and {len(panel_labels)}")
    
    # Load the full wiring data
    wiring_data = pd.read_csv(wiring_csv)
    
    # Validate that all weight columns exist
    missing_cols = [col for col in weight_columns if col not in wiring_data.columns]
    if missing_cols:
        raise ValueError(f"Missing weight columns in wiring CSV: {missing_cols}. "
                        f"Available: {list(wiring_data.columns)}")
    
    # Load modulatory if provided
    modulatory = None
    if modulation_csv and Path(modulation_csv).exists():
        modulatory = pd.read_csv(modulation_csv)
    
    # Get all unique neuron IDs
    all_neurons = set(wiring_data['src'].unique()) | set(wiring_data['tgt'].unique())
    
    # Auto-generate positions if not provided
    if neuron_positions is None:
        n = len(all_neurons)
        neuron_positions = {}
        for i, nid in enumerate(sorted(all_neurons)):
            angle = 2 * np.pi * i / n
            neuron_positions[nid] = (np.cos(angle), np.sin(angle))
    
    # Auto-infer neuron types if not provided
    if neuron_types is None:
        neuron_types = {}
        sources = set(wiring_data['src'].unique())
        targets = set(wiring_data['tgt'].unique())
        
        for nid in all_neurons:
            if nid in sources and nid in targets:
                neuron_types[nid] = 'hidden'
            elif nid in sources:
                neuron_types[nid] = 'input'
            elif nid in targets:
                neuron_types[nid] = 'output'
            else:
                neuron_types[nid] = 'hidden'
    
    # Build neurons dict
    neurons = {}
    for nid in all_neurons:
        neurons[nid] = {
            'pos': neuron_positions[nid],
            'type': neuron_types.get(nid, 'hidden')
        }
    
    # Create figure with subplots (one per panel)
    n_panels = len(weight_columns)
    
    # Determine grid layout (try to make it roughly square)
    if n_panels <= 2:
        ncols = n_panels
        nrows = 1
    elif n_panels <= 4:
        ncols = 2
        nrows = 2
    elif n_panels <= 6:
        ncols = 3
        nrows = 2
    else:
        ncols = min(4, n_panels)
        nrows = (n_panels + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    
    # Ensure axes is always a list (even if only 1 subplot)
    if n_panels == 1:
        axes = [axes]
    else:
        axes = axes.flatten().tolist()
    
    # Draw network in each panel
    axes_list = []
    for ax, weight_col, label in zip(axes[:n_panels], weight_columns, panel_labels):
        # Create connections dataframe with the appropriate weight column
        connections = wiring_data[['src', 'tgt', weight_col]].copy()
        connections = connections.rename(columns={weight_col: 'weight'})
        
        # Draw the network
        draw_network(
            neurons=neurons,
            connections=connections,
            modulatory=modulatory,
            ax=ax,
            title=label,
            show_weights=False,
            weight_column='weight'
        )
        
        # Remove the individual legend from this subplot
        if ax.get_legend():
            ax.get_legend().remove()
        
        axes_list.append(ax)
    
    # Hide any unused subplots
    for ax in axes[n_panels:]:
        ax.axis('off')
    
    # Add a single shared legend at the end
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['neuron_fill'], edgecolor=COLORS['input_edge'], 
                      linewidth=2, label='Input Neuron'),
        mpatches.Patch(facecolor=COLORS['neuron_fill'], edgecolor=COLORS['output_edge'], 
                      linewidth=2, label='Output Neuron'),
        mpatches.Patch(facecolor=COLORS['neuron_fill'], edgecolor=COLORS['hidden_edge'], 
                      linewidth=2, label='Hidden Neuron'),
        plt.Line2D([0], [0], color=COLORS['excitatory'], linewidth=2, 
                   linestyle='-', label='Excitatory Connection'),
        plt.Line2D([0], [0], color=COLORS['inhibitory'], linewidth=2,
                   linestyle='-', label='Inhibitory Connection'),
    ]
    
    # Add the shared legend below the subplots
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.05), ncol=7, fontsize=10, frameon=True, fancybox=True)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.96])
    
    return fig, axes_list


def _get_system_font(size: int = 12):
    """
    Load a system font with fallback options that work cross-platform.
    
    Args:
        size: Font size in points
    
    Returns:
        PIL Font object or None if no suitable font found
    """
    from PIL import ImageFont
    import platform
    
    font_candidates = []
    system = platform.system()
    
    # Platform-specific font paths
    if system == "Windows":
        font_candidates = [
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\calibri.ttf",
        ]
    elif system == "Darwin":  # macOS
        font_candidates = [
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    else:  # Linux and others
        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    
    # Try each candidate
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue
    
    # If no truetype font found, return None to use default
    return None


def draw_and_combine_networks(wiring_csv: str,
                              weight_columns: List[str],
                              panel_labels: List[str],
                              output_path: str,
                              modulation_csv: Optional[str] = None,
                              neuron_positions: Optional[Dict[int, Tuple[float, float]]] = None,
                              neuron_types: Optional[Dict[int, str]] = None,
                              title: str = '') -> str:
    """
    Draw neural networks as separate full-size figures, then combine them into a grid image.
    
    Creates individual high-quality figures for each network, then stitches them together
    into a single grid image with a shared legend.
    
    Args:
        wiring_csv: Path to wiring CSV file
        weight_columns: List of weight column names to plot
        panel_labels: List of labels for each panel
        output_path: Path where to save the combined image
        modulation_csv: Path to modulation CSV (optional)
        neuron_positions: Manual positions for neurons
        neuron_types: Dict mapping neuron_id -> type
        title: Overall title for the combined image
    
    Returns:
        Path to the saved combined image
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError("Pillow is required for combining network images. Install with: pip install Pillow")
    
    if len(weight_columns) != len(panel_labels):
        raise ValueError(f"weight_columns and panel_labels must have same length")
    
    # Load the full wiring data
    wiring_data = pd.read_csv(wiring_csv)
    
    # Validate columns exist
    missing_cols = [col for col in weight_columns if col not in wiring_data.columns]
    if missing_cols:
        raise ValueError(f"Missing weight columns: {missing_cols}")
    
    # Load modulatory if provided
    modulatory = None
    if modulation_csv and Path(modulation_csv).exists():
        modulatory = pd.read_csv(modulation_csv)
    
    # Get all unique neuron IDs
    all_neurons = set(wiring_data['src'].unique()) | set(wiring_data['tgt'].unique())
    
    # Auto-generate positions if not provided
    if neuron_positions is None:
        n = len(all_neurons)
        neuron_positions = {}
        for i, nid in enumerate(sorted(all_neurons)):
            angle = 2 * np.pi * i / n
            neuron_positions[nid] = (np.cos(angle), np.sin(angle))
    
    # Auto-infer neuron types if not provided
    if neuron_types is None:
        neuron_types = {}
        sources = set(wiring_data['src'].unique())
        targets = set(wiring_data['tgt'].unique())
        
        for nid in all_neurons:
            if nid in sources and nid in targets:
                neuron_types[nid] = 'hidden'
            elif nid in sources:
                neuron_types[nid] = 'input'
            elif nid in targets:
                neuron_types[nid] = 'output'
            else:
                neuron_types[nid] = 'hidden'
    
    # Build neurons dict
    neurons = {}
    for nid in all_neurons:
        neurons[nid] = {
            'pos': neuron_positions[nid],
            'type': neuron_types.get(nid, 'hidden')
        }
    
    # Create individual full-size figures and save them
    individual_images = []
    for weight_col, label in zip(weight_columns, panel_labels):
        # Create connections dataframe
        connections = wiring_data[['src', 'tgt', weight_col]].copy()
        connections = connections.rename(columns={weight_col: 'weight'})
        
        # Create a full-size figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw the network
        draw_network(
            neurons=neurons,
            connections=connections,
            modulatory=modulatory,
            ax=ax,
            title=label,
            show_weights=False,
            weight_column='weight'
        )
        
        # Remove the legend from individual figures (will add shared one later)
        if ax.get_legend():
            ax.get_legend().remove()
        
        plt.tight_layout()
        
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        fig.savefig(tmp_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        individual_images.append(tmp_path)
    
    # Combine images into a grid
    n_images = len(individual_images)
    ncols = 2
    nrows = (n_images + ncols - 1) // ncols
    
    # Load all images
    pil_images = [Image.open(img_path) for img_path in individual_images]
    
    # Get dimensions (assuming all images are same size)
    img_width, img_height = pil_images[0].size
    
    # Create combined image with padding and title space
    grid_width = img_width * ncols
    grid_height = img_height * nrows
    title_height = 140 if title else 0
    legend_height = 140  # Space for legend below (increased for proper sizing)
    spacing = 20
    
    combined_width = grid_width + spacing * (ncols + 1)
    combined_height = grid_height + spacing * (nrows + 1) + title_height + legend_height
    
    combined_img = Image.new('RGB', (combined_width, combined_height), color='white')
    
    # Paste images into grid with spacing
    border_width = 3
    for idx, pil_img in enumerate(pil_images):
        row = idx // ncols
        col = idx % ncols
        x = col * (img_width + spacing) + spacing
        y = row * (img_height + spacing) + title_height + spacing
        combined_img.paste(pil_img, (x, y))
        
        # Draw border around the image
        draw = ImageDraw.Draw(combined_img)
        border_color = 'black'
        draw.rectangle(
            [x - border_width, y - border_width, 
             x + img_width + border_width, y + img_height + border_width],
            outline=border_color,
            width=border_width
        )
    
    # Add title if provided
    if title:
        draw = ImageDraw.Draw(combined_img)
        # Try to load a system font with fallback to default
        font = _get_system_font(size=104)
        if font is None:
            font = _get_system_font(size=80)
        if font is None:
            font = ImageFont.load_default()
        
        # Calculate centered title position
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (combined_width - title_width) // 2
        draw.text((title_x, 15), title, fill='black', font=font)
    
    # Add legend below the plots in a horizontal line
    legend_y = grid_height + spacing * (nrows + 1) + title_height + 20
    draw = ImageDraw.Draw(combined_img)
    font_small = _get_system_font(size=56)
    if font_small is None:
        font_small = _get_system_font(size=48)
    if font_small is None:
        font_small = ImageFont.load_default()
    
    # Convert color hex strings to RGB tuples for PIL
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    neuron_fill_rgb = hex_to_rgb(COLORS['neuron_fill'])
    input_edge_rgb = hex_to_rgb(COLORS['input_edge'])
    output_edge_rgb = hex_to_rgb(COLORS['output_edge'])
    hidden_edge_rgb = hex_to_rgb(COLORS['hidden_edge'])
    excitatory_rgb = hex_to_rgb(COLORS['excitatory'])
    inhibitory_rgb = hex_to_rgb(COLORS['inhibitory'])
    
    # Legend items with their types: ('label', type, color1, color2)
    # type can be 'neuron' (fill, edge), 'line' (color, style)
    legend_items = [
        ("Input", 'neuron', neuron_fill_rgb, input_edge_rgb),
        ("Output", 'neuron', neuron_fill_rgb, output_edge_rgb),
        ("Hidden", 'neuron', neuron_fill_rgb, hidden_edge_rgb),
        ("Excitatory", 'line', excitatory_rgb, None),
        ("Inhibitory", 'line', inhibitory_rgb, None),
    ]
    
    swatch_size = 50
    swatch_spacing = 15
    line_thickness = 6
    line_length = 60
    x_pos = spacing + 20
    
    for label, item_type, color1, color2 in legend_items:
        # Calculate vertical center for this item
        text_bbox = draw.textbbox((0, 0), label, font=font_small)
        text_height = text_bbox[3] - text_bbox[1]
        y_center_offset = (text_height - swatch_size) // 2
        
        # Draw the swatch
        if item_type == 'neuron':
            # Draw filled square with edge
            draw.rectangle(
                [x_pos, legend_y + y_center_offset, x_pos + swatch_size, legend_y + y_center_offset + swatch_size],
                fill=color1,
                outline=color2,
                width=6
            )
        elif item_type == 'line':
            # Draw solid line (longer and thicker)
            y_center = legend_y + y_center_offset + swatch_size // 2
            draw.line(
                [(x_pos, y_center), (x_pos + line_length, y_center)],
                fill=color1,
                width=line_thickness
            )
        elif item_type == 'dashed_line':
            # Draw dashed line (longer and thicker)
            y_center = legend_y + y_center_offset + swatch_size // 2
            dash_length = 6
            gap_length = 4
            for x in range(x_pos, x_pos + line_length, dash_length + gap_length):
                draw.line(
                    [(x, y_center), (min(x + dash_length, x_pos + line_length), y_center)],
                    fill=color1,
                    width=line_thickness
                )
        
        # Draw label
        text_x = x_pos + max(swatch_size, line_length) + swatch_spacing
        draw.text((text_x, legend_y + y_center_offset), label, fill='black', font=font_small)
        
        # Move to next position
        text_bbox = draw.textbbox((text_x, legend_y + y_center_offset), label, font=font_small)
        text_width = text_bbox[2] - text_bbox[0]
        x_pos = text_x + text_width + 50  # 50 pixels spacing between items
    
    # Save the combined image
    combined_img.save(output_path)
        
    # Clean up temporary files
    import os
    for tmp_path in individual_images:
        try:
            os.remove(tmp_path)
        except:
            pass
    
    return output_path


# ===== Utility Functions for Loading Data =====

def load_network_from_csvs(wiring_csv: str,
                          modulation_csv: Optional[str] = None,
                          neuron_positions: Optional[Dict[int, Tuple[float, float]]] = None,
                          neuron_types: Optional[Dict[int, str]] = None,
                          weight_column: str = 'weight_initial',
                          auto_layout: bool = False,
                          use_graphviz: bool = True,
                          spring_k: float = 2.0) -> Tuple[Dict, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load network data from CSV files.
    
    Args:
        wiring_csv: Path to wiring CSV (must have 'src', 'tgt', and weight columns)
        modulation_csv: Path to modulation CSV (optional)
        neuron_positions: Manual positions for neurons. If None and auto_layout=False, 
                         auto-generates a circular layout
        neuron_types: Dict mapping neuron_id -> type. If None, infers from data
        weight_column: Which weight column to use from wiring CSV
        auto_layout: If True, compute positions automatically from network structure
                    using graphviz (dot) or spring layout as fallback
        use_graphviz: If True (with auto_layout=True), try graphviz first
        spring_k: Repulsive force for spring layout (larger = more spread)
    
    Returns:
        (neurons, connections, modulatory) tuple ready for draw_network()
    
    Example:
        # Auto-compute layout from network structure (no YAML config needed)
        neurons, connections, modulatory = load_network_from_csvs(
            'path/to/wiring.csv',
            auto_layout=True  # This does the magic!
        )
        draw_network(neurons, connections, modulatory)
    """
    # Load connections
    connections = pd.read_csv(wiring_csv)
    
    # Load modulatory if provided
    modulatory = None
    if modulation_csv:
        modulatory = pd.read_csv(modulation_csv)
    
    # Get all unique neuron IDs
    all_neurons = set(connections['src'].unique()) | set(connections['tgt'].unique())
    
    # Compute or use provided positions
    if auto_layout:
        neuron_positions = compute_layout(connections, neuron_types=neuron_types, 
                                         use_graphviz=use_graphviz, spring_k=spring_k)
        print(f"Auto-computed positions for {len(neuron_positions)} neurons")
    elif neuron_positions is None:
        # Default: circular layout
        n = len(all_neurons)
        neuron_positions = {}
        for i, nid in enumerate(sorted(all_neurons)):
            angle = 2 * np.pi * i / n
            neuron_positions[nid] = (np.cos(angle), np.sin(angle))
    
    # Auto-infer neuron types if not provided
    if neuron_types is None:
        neuron_types = {}
        # Simple heuristic: neurons with src but no tgt might be input,
        # neurons with tgt but no src might be output
        sources = set(connections['src'].unique())
        targets = set(connections['tgt'].unique())
        
        for nid in all_neurons:
            if nid in sources and nid in targets:
                neuron_types[nid] = 'hidden'
            elif nid in sources:
                neuron_types[nid] = 'input'
            elif nid in targets:
                neuron_types[nid] = 'output'
            else:
                neuron_types[nid] = 'hidden'
    
    # Build neurons dict
    neurons = {}
    for nid in all_neurons:
        neurons[nid] = {
            'pos': neuron_positions[nid],
            'type': neuron_types.get(nid, 'hidden')
        }
    
    # Select the appropriate weight column
    if weight_column in connections.columns:
        connections = connections.rename(columns={weight_column: 'weight'})
    
    return neurons, connections, modulatory