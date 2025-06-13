import logging
logger = logging.getLogger(__name__)

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def detect_k_recommendations(silhouette_scores: dict, threshold: float = 0.5):
    """
    Detects the peak and shoulder k-values from silhouette scores.
    
    Parameters:
        silhouette_scores (dict): Dictionary {k: silhouette_score}/
        threshold (float): Minimum silhouette score considered acceptable for 'shoulder' value

    Returns:
        tuple: (peak_k, shoulder_k)
    """

    # filter valid silhouette scores
    valid_scores = {k: score for k, score in silhouette_scores.items() if score is not None and not np.isnan(score)}

    if not valid_scores:
        return None, None
    
    # Peak: k with highest silhouette
    peak_k = max(valid_scores, key=valid_scores.get)

    # Shoulder: highest k with silhouette score above threshold 
    shoulder_candidates = [k for k, score in valid_scores.items() if score > threshold]
    shoulder_k = max(shoulder_candidates) if shoulder_candidates else None

    return peak_k, shoulder_k

def plot_kprototypes_results(
    k_range, costs, silhouettes, peak_k=None, shoulder_k=None,
    title='K-Prototypes Evaluation', threshold=0.5, width=1000, height=600
):
    x_vals = list(k_range)
    cost_vals = [costs.get(k, None) for k in x_vals]
    silhouette_vals = [silhouettes.get(k, None) for k in x_vals]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Cost curve
    fig.add_trace(go.Scatter(
        x=x_vals, y=cost_vals, name='Cost', mode='lines+markers', line=dict(color='gray')
    ), secondary_y=False)

    # Silhouette curve
    fig.add_trace(go.Scatter(
        x=x_vals, y=silhouette_vals, name='Silhouette', mode='lines+markers', line=dict(color='#7741EC')
    ), secondary_y=True)

    # Horizontal threshold line
    fig.add_hline(y=threshold, line=dict(color='lightgray', dash='dot'), secondary_y=True)

    # Peak marker
    if peak_k:
        fig.add_trace(go.Scatter(
            x=[peak_k], y=[silhouettes[peak_k]], mode='markers+text',
            text=[f'Peak (k={peak_k})'], textposition='top center',
            marker=dict(size=10, color="#7741EC"), name='Peak'
        ), secondary_y=True)
        fig.add_vline(x=peak_k, line=dict(dash='dash', color='#7741EC'))


    # Shoulder marker
    if shoulder_k:
        fig.add_trace(go.Scatter(
            x=[shoulder_k], y=[silhouettes[shoulder_k]], mode='markers+text',
            text=[f'Shoulder (k={shoulder_k})'], textposition='bottom center',
            marker=dict(size=10, color="#7741EC"), name='Shoulder'
        ), secondary_y=True)
        fig.add_vline(x=shoulder_k, line=dict(dash='dash'))

    # Layout and formatting
    fig.update_layout(
        title=title,
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Cost (WSS)',
        yaxis2_title='Silhouette Score',
        template='plotly',  # <-- Better for dark mode
        width=width,
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(showgrid=False, secondary_y=False)
    fig.update_yaxes(showgrid=True, secondary_y=True)

    return fig
