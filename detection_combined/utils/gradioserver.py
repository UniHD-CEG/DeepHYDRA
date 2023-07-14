import math

import pandas as pd

import gradio as gr
import datetime
import numpy as np

plot_end = 2*math.pi

class GradioServer():
    def __init__(self) -> None:

        self._demo = gr.Blocks(analytics_enabled=False)

        with self._demo:
            with gr.Row():
                with gr.Column():
                    c_time2 = gr.Textbox(label="Current Time refreshed every second")
                    gr.Textbox(
                        "Change the value of the slider to automatically update the plot",
                        label="",
                    )
                    period = gr.Slider(
                        label="Period of plot", value=1, minimum=0, maximum=10, step=1
                    )
                    plot = gr.LinePlot(show_label=False)
                with gr.Column():
                    name = gr.Textbox(label="Enter your name")
                    greeting = gr.Textbox(label="Greeting")
                    button = gr.Button(value="Greet")
                    button.click(lambda s: f"Hello {s}", name, greeting)

            self._demo.load(lambda: datetime.datetime.now(), None, c_time2, every=1)
            dep = self._demo.load(self.get_plot, None, plot, every=1)
            period.change(self.get_plot, period, plot, every=1, cancels=[dep])


    def get_plot(self, period=1):
        global plot_end
        x = np.arange(plot_end - 2*math.pi, plot_end, 0.02)
        y = np.sin(2*math.pi*period*x)
        update = gr.LinePlot.update(
            value=pd.DataFrame({"x": x, "y": y}),
            x="x",
            y="y",
            title="Plot (updates every second)",
            width=600,
            height=350,
        )
        plot_end += 2*math.pi
        if plot_end > 1000:
            plot_end = 2*math.pi
        return update
    

    def launch(self):
        self._demo.queue().launch()

