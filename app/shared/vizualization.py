import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import altair as alt

import altair as alt
import pandas as pd

def visualize_column_data(dtype: str, column_data: pd.Series, col_name: str) -> alt.Chart:
    column_data = column_data.dropna()

    if dtype == "categorical":
        counts = column_data.value_counts().reset_index()
        counts.columns = ["value", "count"]
        counts["column"] = col_name

        chart = alt.Chart(counts).mark_bar().encode(
            x=alt.X("count:Q", stack="zero", title=None),
            y=alt.Y("column:N", title=None, axis=alt.Axis(labels=False)),
            color=alt.Color("value:N",
                sort=alt.EncodingSortField(field="count", op="sum", order="descending"),
                legend=None
            ),
            tooltip=["value:N", "count:Q"]
        ).properties(height=65)

    elif dtype == "numerical":
        df = pd.DataFrame({"value": column_data, "column": col_name})
        chart = alt.Chart(df).mark_boxplot(extent="min-max", ticks=False).encode(
            x=alt.X("value:Q", title=None),
            y=alt.Y("column:N", title=None, axis=alt.Axis(labels=False))
        ).properties(height=75)

    else:
        # Return an invisible placeholder
        chart = alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_point()

    return chart.configure_view(stroke=None)


def plot_kprototypes_results(
    k_range, costs, silhouettes, peak_k=None, shoulder_k=None,
    title='K-Prototypes Evaluation', threshold=0.5, width=400, height=400
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
        x=x_vals, y=silhouette_vals, name='Silhouette', mode='lines+markers', line=dict(color='#1471b0')
    ), secondary_y=True)

    # Horizontal threshold line
    fig.add_hline(y=threshold, line=dict(color='lightgray', dash='dot'), secondary_y=True)

    # Peak marker
    if peak_k:
        fig.add_trace(go.Scatter(
            x=[peak_k], y=[silhouettes[peak_k]], mode='markers+text',
            text=[f'Peak (k={peak_k})'], textposition='top center',
            marker=dict(size=10, color="#1471b0"), name='Peak'
        ), secondary_y=True)
        fig.add_vline(x=peak_k, line=dict(dash='dash', color='#1471b0'))


    # Shoulder marker
    if shoulder_k:
        fig.add_trace(go.Scatter(
            x=[shoulder_k], y=[silhouettes[shoulder_k]], mode='markers+text',
            text=[f'Shoulder (k={shoulder_k})'], textposition='bottom center',
            marker=dict(size=10, color="#1471b0"), name='Shoulder'
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