from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np
import os

from molmap.utils.logtools import print_info

try:
    from highcharts_core.chart import Chart
except Exception:
    Chart = None


def _require_highcharts_core():
    if Chart is None:
        raise ImportError(
            "highcharts-core is required for HTML visualization. "
            "Please install it with: pip install highcharts-core"
        )


def _save_chart(chart, filename: str):
    """
    Save a highcharts-core chart to HTML.
    """
    filename = str(filename)
    if filename.endswith(".html"):
        html_file = filename
    else:
        html_file = filename + ".html"

    try:
        chart.save_chart(filename=html_file)
    except Exception:
        # Fallback for versions exposing different APIs
        try:
            html = chart.to_html()
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html)
        except Exception as e:
            raise RuntimeError(f"Failed to save chart to {html_file}: {e}")

    return html_file


def _make_scatter_chart(title, subtitle, radius=2, enabled_data_labels=False):
    _require_highcharts_core()

    chart = Chart.from_options({
        "chart": {"type": "scatter", "zoomType": "xy"},
        "title": {"text": title},
        "subtitle": {"text": subtitle},
        "xAxis": {
            "title": {"text": "X", "style": {"fontSize": 20}},
            "labels": {"style": {"fontSize": 20}},
            "gridLineWidth": 1,
            "startOnTick": True,
            "endOnTick": True,
            "showLastLabel": True,
        },
        "yAxis": {
            "title": {"text": "Y", "style": {"fontSize": 20}},
            "labels": {"style": {"fontSize": 20}},
            "gridLineWidth": 1,
        },
        "legend": {
            "align": "right",
            "layout": "vertical",
            "margin": 1,
            "verticalAlign": "top",
            "y": 40,
            "symbolHeight": 12,
            "floating": False,
        },
        "plotOptions": {
            "scatter": {
                "marker": {
                    "radius": radius,
                    "states": {
                        "hover": {
                            "enabled": True,
                            "lineColor": "rgb(100,100,100)"
                        }
                    },
                },
                "states": {"hover": {"marker": {"enabled": False}}},
                "tooltip": {
                    "headerFormat": "<b>{series.name}</b><br>",
                    "pointFormat": "{point.IDs}",
                },
            },
            "series": {
                "turboThreshold": 5000,
                "dataLabels": {
                    "enabled": enabled_data_labels,
                    "format": "{point.IDs}",
                },
            },
        },
        "series": [],
    })
    return chart


def _make_heatmap_chart(title, subtitle, mp, enabled_data_labels=False):
    _require_highcharts_core()

    chart = Chart.from_options({
        "chart": {"type": "heatmap", "zoomType": "xy"},
        "title": {"text": title},
        "subtitle": {"text": subtitle},
        "xAxis": {
            "title": None,
            "min": 0,
            "max": mp.fmap_shape[1] - 1,
            "startOnTick": False,
            "endOnTick": False,
            "allowDecimals": False,
            "labels": {"style": {"fontSize": 20}},
        },
        "yAxis": {
            "title": {"text": " ", "style": {"fontSize": 20}},
            "startOnTick": False,
            "endOnTick": False,
            "gridLineWidth": 0,
            "reversed": True,
            "min": 0,
            "max": mp.fmap_shape[0] - 1,
            "allowDecimals": False,
            "labels": {"style": {"fontSize": 20}},
        },
        "legend": {
            "align": "right",
            "layout": "vertical",
            "margin": 1,
            "verticalAlign": "top",
            "y": 60,
            "symbolHeight": 12,
            "floating": False,
        },
        "tooltip": {
            "headerFormat": "<b>{series.name}</b><br>",
            "pointFormat": "{point.v}",
        },
        "plotOptions": {
            "series": {
                "turboThreshold": 5000,
                "dataLabels": {
                    "enabled": enabled_data_labels,
                    "format": "{point.v}",
                    "style": {"textOutline": False, "color": "black"},
                },
            }
        },
        "series": [],
    })
    return chart


def plot_scatter(mp, htmlpath="./", htmlname=None, radius=2, enabled_data_labels=False):
    """
    mp: the object of mp
    htmlpath: the figure path, not include the prefix of 'html'
    htmlname: the name
    radius: int, default: 2, the radius of scatter dot
    """
    title = f"2D embedding of {mp.ftype} based on {mp.emb_method} method"
    subtitle = f"number of {mp.ftype}: {len(mp.flist)}, metric method: {mp.metric}"
    name = f"{mp.ftype}_{len(mp.flist)}_{mp.metric}_{mp.emb_method}_scatter"

    if not os.path.exists(htmlpath):
        os.makedirs(htmlpath)

    if htmlname:
        name = f"{htmlname}_{name}"

    filename = os.path.join(htmlpath, name)
    print_info(f"generate file: {filename}")

    xy = mp.embedded.embedding_
    colormaps = mp.colormaps

    df = pd.DataFrame(xy, columns=["x", "y"])
    bitsinfo = mp.bitsinfo.set_index("IDs")
    df = df.join(bitsinfo.loc[mp.flist].reset_index())
    df["colors"] = df["Subtypes"].map(colormaps)

    chart = _make_scatter_chart(
        title=title,
        subtitle=subtitle,
        radius=radius,
        enabled_data_labels=enabled_data_labels,
    )

    series = []
    for subtype, color in colormaps.items():
        dfi = df[df["Subtypes"] == subtype]
        if len(dfi) == 0:
            continue

        data = dfi[["x", "y", "IDs"]].to_dict("records")
        series.append({
            "type": "scatter",
            "name": subtype,
            "color": color,
            "data": data,
        })

    chart.options.series = series
    saved_file = _save_chart(chart, filename)
    print_info(f"save html file to {saved_file}")
    return df, chart


def plot_grid(mp, htmlpath="./", htmlname=None, enabled_data_labels=False):
    """
    mp: the object of mp
    htmlpath: the figure path
    """
    if not os.path.exists(htmlpath):
        os.makedirs(htmlpath)

    title = f"Assignment of {mp.ftype} by {mp.emb_method} embedding result"
    subtitle = f"number of {mp.ftype}: {len(mp.flist)}, metric method: {mp.metric}"
    name = f"{mp.ftype}_{len(mp.flist)}_{mp.metric}_{mp.emb_method}_mp"

    if htmlname:
        name = f"{htmlname}_{name}"

    filename = os.path.join(htmlpath, name)
    print_info(f"generate file: {filename}")

    m, n = mp.fmap_shape
    colormaps = mp.colormaps
    position = np.zeros(mp.fmap_shape, dtype="O").reshape(m * n,)
    position[mp._S.col_asses] = mp.flist
    position = position.reshape(m, n)

    x = []
    for i in range(n):
        x.extend([i] * m)

    y = list(range(m)) * n
    v = position.reshape(m * n, order="f")

    df = pd.DataFrame(list(zip(x, y, v)), columns=["x", "y", "v"])
    bitsinfo = mp.bitsinfo
    subtypedict = bitsinfo.set_index("IDs")["Subtypes"].to_dict()
    subtypedict.update({0: "NaN"})
    df["Subtypes"] = df["v"].map(subtypedict)
    df["colors"] = df["Subtypes"].map(colormaps)

    chart = _make_heatmap_chart(
        title=title,
        subtitle=subtitle,
        mp=mp,
        enabled_data_labels=enabled_data_labels,
    )

    series = []
    for subtype, color in colormaps.items():
        dfi = df[df["Subtypes"] == subtype]
        if len(dfi) == 0:
            continue

        data = dfi[["x", "y", "v"]].to_dict("records")
        series.append({
            "type": "heatmap",
            "name": subtype,
            "color": color,
            "data": data,
        })

    chart.options.series = series
    saved_file = _save_chart(chart, filename)
    print_info(f"save html file to {saved_file}")
    return df, chart


def _getNewick(node, newick, parentdist, leaf_names):
    if node.is_leaf():
        return "%s:%.2f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parentdist - node.dist, newick)
        else:
            newick = ");"
        newick = _getNewick(node.get_left(), newick, node.dist, leaf_names)
        newick = _getNewick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
        newick = "(%s" % (newick)
        return newick


def _mp2newick(mp, treefile="mytree"):
    dist_matrix = mp.dist_matrix
    leaf_names = mp.flist
    df = mp.df_embedding[["colors", "Subtypes"]]

    dists = squareform(dist_matrix)
    linkage_matrix = linkage(dists, "complete")
    tree = to_tree(linkage_matrix, rd=False)
    newick = _getNewick(tree, "", tree.dist, leaf_names=leaf_names)

    with open(treefile + ".nwk", "w", encoding="utf-8") as f:
        f.write(newick)
    df.to_excel(treefile + ".xlsx")


def plot_tree(mp, htmlpath="./", htmlname=None):
    raise NotImplementedError("plot_tree is not implemented yet.")