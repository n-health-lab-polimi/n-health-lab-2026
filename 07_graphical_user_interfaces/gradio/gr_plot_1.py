import gradio as gr
import pandas as pd
import numpy as np
import random

from gr_data import df

with gr.Blocks() as demo:
    gr.LinePlot(df, x="weight", y="height")

demo.launch()
