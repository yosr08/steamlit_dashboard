
import io
from typing import List, Optional

import markdown
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly import express as px
from plotly.subplots import make_subplots
import pydeck as pdk
from PIL import Image
import seaborn as sns
from datetime import datetime
DATA_URL = (
    'C:/Users/Lenovo/export_dataframe.csv'
)


# matplotlib.use("TkAgg")
matplotlib.use("Agg")
COLOR = "black"
BACKGROUND_COLOR = "#fff"

def main():
    """Main function. Run this to run the app"""
    st.sidebar.title("smart bins collection")
    image = Image.open('cc.jpg')
    imge = st.sidebar.image(image,use_column_width=True)
    now = datetime.now()
    st.sidebar.title("Today date")
    st.sidebar.write(now.strftime("%d/%m/%Y %H:%M:%S"))
    st.sidebar.header("Settings")
    image = Image.open('delete.png')
    st.image(image, width=300)
    st.markdown(
        """
# Smart bins collection
"this application is a streamlit app that can be used
             to analyse bins collection 🗑🚮"
"""
    )

    @st.cache(persist = True)
    def load_data():
        data = pd.read_csv(DATA_URL , parse_dates=['timestamp'] )
        data.dropna(subset=['latitude', 'longitude'], inplace=True)
        data['year'] = data['timestamp'].dt.year.astype('uint8')
        data['month'] = data['timestamp'].dt.month.astype('uint8')
        data['day'] = data['timestamp'].dt.day.astype('uint8')
        return data

    data = load_data()
    data[['latitude','longitude']].to_csv('lat_long.csv', index=False)

    midpoint = (np.average(data["latitude"]), np.average(data["longitude"]))
    st.markdown(""" # Map representing the location of our bins and level of fillness""")

    st.write(px.scatter_mapbox(data, lat="latitude", lon="longitude", color="fullnessThreshold" , size="bin_number" ,
                  color_continuous_scale=px.colors.sequential.Redor, size_max=15, zoom=10 ,animation_frame="day",
                  mapbox_style="carto-positron"))

    st.markdown(""" # Top 5 full bins""")
    new_data = data.query("fullnessThreshold >= 1")[["bin_number", "fullnessThreshold"]].sort_values(by=['fullnessThreshold'], ascending=False).dropna(how="any")[:5]
    st.write(px.pie(new_data, values='fullnessThreshold', names = "bin_number",color_discrete_sequence=px.colors.sequential.RdBu))

    st.markdown(""" # different fill levels of the bins""")
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [1, 2, 3])
    sns.countplot(x="fullnessThreshold", data= data ,palette="pastel")
    st.pyplot(fig)

    st.markdown(""" # Specific informations about the bins""")
    st.sidebar.subheader("Bins information")
    select = st.sidebar.slider("choose bin number", 1 , 500)
    modified_data = data[data['bin_number'] == select]
    ax.scatter([1, 2, 3], [1, 2, 3])
    sns.countplot(x="fullnessThreshold", data= modified_data ,palette="pastel").set(title='fillness level')
    st.sidebar.pyplot(fig)


    # My preliminary idea of an API for generating a grid
    with Grid("1 1 1", color= '#ff9966', background_color=BACKGROUND_COLOR) as grid:
        grid.cell(
            class_="a",
            grid_column_start=1,
            grid_column_end=3,
            grid_row_start=0,
            grid_row_end=2,
        ).markdown("# bin %i informations 📊" %select)
        volume = modified_data['fullnessThreshold']
        grid.cell("b", 1, 3, 2, 3).text("""
        FILL LEVEL:
         %i
         """ %volume)
        status= modified_data['status']
        grid.cell("e", 1, 3, 3, 4).text("""
        STATUS OF THE BIN:
        %s
        """ %status )



class Cell:
    """A Cell can hold text, markdown, plots etc."""

    def __init__(
        self,
        class_: str = None,
        grid_column_start: Optional[int] = None,
        grid_column_end: Optional[int] = None,
        grid_row_start: Optional[int] = None,
        grid_row_end: Optional[int] = None,
    ):
        self.class_ = class_
        self.grid_column_start = grid_column_start
        self.grid_column_end = grid_column_end
        self.grid_row_start = grid_row_start
        self.grid_row_end = grid_row_end
        self.inner_html = ""

    def _to_style(self) -> str:
        return f"""
.{self.class_} {{
    grid-column-start: {self.grid_column_start};
    grid-column-end: {self.grid_column_end};
    grid-row-start: {self.grid_row_start};
    grid-row-end: {self.grid_row_end};
}}
"""

    def text(self, text: str = ""):
        self.inner_html = text

    def markdown(self, text):
        self.inner_html = markdown.markdown(text)


    def dataframe(self, data):
        self.inner_html = dataframe.to_html()

    def plotly_chart(self, fig):
        self.inner_html = f"""
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<body>
<p>This should have been a plotly plot.
But since *script* tags are removed when inserting MarkDown/ HTML i cannot get it to workto work.
But I could potentially save to svg and insert that.</p>
<div id='divPlotly'></div>
<script>
    var plotly_data = {fig.to_json()}
    Plotly.react('divPlotly', plotly_data.data, plotly_data.layout);
</script>
</body>
"""

    def pyplot(self, fig=None, **kwargs):
        string_io = io.StringIO()
        plt.savefig(string_io, format="svg", fig=(6, 4))
        svg = string_io.getvalue()[215:]
        plt.close(fig)
        self.inner_html = '<div height="200px">' + svg + "</div>"

    def _to_html(self):
        return f"""<div class="box {self.class_}">{self.inner_html}</div>"""


class Grid:
    """A (CSS) Grid"""

    def __init__(
        self,
        template_columns="1 1 1",
        gap="10px",
        background_color=COLOR,
        color=BACKGROUND_COLOR,
    ):
        self.template_columns = template_columns
        self.gap = gap
        self.background_color = background_color
        self.color = color
        self.cells: List[Cell] = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        st.markdown(self._get_grid_style(), unsafe_allow_html=True)
        st.markdown(self._get_cells_style(), unsafe_allow_html=True)
        st.markdown(self._get_cells_html(), unsafe_allow_html=True)

    def _get_grid_style(self):
        return f"""
<style>
    .wrapper {{
    display: grid;
    grid-template-columns: {self.template_columns};
    grid-gap: {self.gap};
    background-color: {self.background_color};
    color: {self.color};
    }}
    .box {{
    background-color: {self.color};
    color: {self.background_color};
    border-radius: 5px;
    padding: 20px;
    font-size: 150%;
    }}
    table {{
        color: {self.color}
    }}
</style>
"""

    def _get_cells_style(self):
        return (
            "<style>"
            + "\n".join([cell._to_style() for cell in self.cells])
            + "</style>"
        )

    def _get_cells_html(self):
        return (
            '<div class="wrapper">'
            + "\n".join([cell._to_html() for cell in self.cells])
            + "</div>"
        )

    def cell(
        self,
        class_: str = None,
        grid_column_start: Optional[int] = None,
        grid_column_end: Optional[int] = None,
        grid_row_start: Optional[int] = None,
        grid_row_end: Optional[int] = None,
    ):
        cell = Cell(
            class_=class_,
            grid_column_start=grid_column_start,
            grid_column_end=grid_column_end,
            grid_row_start=grid_row_start,
            grid_row_end=grid_row_end,
        )
        self.cells.append(cell)
        return cell



@st.cache
def get_dataframe(nrows):
    data = pd.read_csv(DATA_URL , nrows=nrows , parse_dates=['timestamp'] )
    data.dropna(subset=['latitude', 'longitude'], inplace=True)
    return data


def get_plotly_fig(data):
    """Dummy Plotly Plot"""
    new_data = data.query("fullnessThreshold >= 1")[["bin_number", "fullnessThreshold"]].sort_values(by=['fullnessThreshold'], ascending=False).dropna(how="any")[:5]
    fig = px.pie(new_data, values='fullnessThreshold', title='Top 5 full bins')
    return fig





main()
