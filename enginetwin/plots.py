import plotly.graph_objects as go

from enginetwin.efficiency_map import sweet_spot


def efficiency_heatmap(cells, source: str) -> go.Figure:
    """Efficiency island over rpm x torque with the sweet spot marked. `source` is real / simulated / mixed."""
    grid = cells.pivot(index="torque", columns="rpm", values="efficiency")
    best = sweet_spot(cells)

    fig = go.Figure(
        [
            go.Heatmap(
                x=grid.columns,
                y=grid.index,
                z=grid.values,
                colorscale="Viridis",
                colorbar={"title": "Efficiency", "tickformat": ".0%"},
                hovertemplate="%{x} rpm, %{y} Nm: %{z:.1%}<extra></extra>",
            ),
            go.Scatter(
                x=[best["rpm"]],
                y=[best["torque"]],
                mode="markers+text",
                marker={"symbol": "star", "size": 16, "color": "white", "line": {"color": "black", "width": 1}},
                text=[f"sweet spot {best['efficiency']:.1%}"],
                textposition="top center",
                name="Sweet spot",
            ),
        ]
    )
    fig.update_layout(
        title=f"Engine efficiency map ({source.capitalize()} data)",
        xaxis_title="Engine speed (rpm)",
        yaxis_title="Engine torque (Nm)",
    )
    return fig
