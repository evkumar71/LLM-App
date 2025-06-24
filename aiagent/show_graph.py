# pip install ipython

from IPython.display import Image, display
import PIL.Image
from io import BytesIO
from langgraph.graph import StateGraph


def show_graph(graph: StateGraph):
    try:
        img = Image(graph.get_graph().draw_mermaid_png())
        pimg = PIL.Image.open(BytesIO(img.data))
        pimg.show()

    except Exception:
        # This requires some extra dependencies and is optional
        pass
