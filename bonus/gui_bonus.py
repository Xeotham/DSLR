#!../.venv/bin/python
import sys
sys.path.insert(1, "..")
sys.path.insert(2, ".")

from subprocess import run
from pandas import read_csv
from customtkinter import *
import customtkinter as ctk
from CTkTable import *
from os import devnull
from re import sub
from contextlib import redirect_stdout

from builders.button_builder import ButtonBuilder
from dslr_lib.errors import print_error
from analysis.describe import describe
from dslr_lib.threads_bonus import threaded

ctk.DrawEngine.preferred_drawing_method = "circle_shapes"
app = CTk()
button_builder = ButtonBuilder(app)

HAS_DESCRIBED = False
WIDTH=700
HEIGHT=200
CORNER_RADIUS=5
COURSES = [
    "Arithmancy",
    "Astronomy",
    "Herbology",
    "Defense Against the Dark Arts",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    "Charms",
    "Flying",
    "All"
]

def execute_script(path: str, args: list[str] | None = None):
    python_interpreter = str(sys.executable)
    absolute_path = __file__.replace("/bonus/./gui_bonus.py", "")
    absolute_path += path

    command = [str(python_interpreter), str(absolute_path)]
    if args:
        for arg in args:
            command.append(str(arg))
    run(command)


def create_exit_button():
    button_builder.new()                \
    .text("Close")                      \
    .size(40, 28)                       \
    .fg_color("#a51f1f")                \
    .hover(True, "#701414")             \
    .command(lambda: app.destroy())     \
    .pack(
        side="right",
        padx=5,
        pady=5,
        anchor="n"
    )


def create_tabs():
    def tabview_callback():
        global HAS_DESCRIBED
        match tabview.get():
            case "Analysis":
                if HAS_DESCRIBED:
                    app.geometry(f"{WIDTH}x800")
                else:
                    app.geometry(f"{WIDTH}x{HEIGHT}")
            case "Visuals":
                app.geometry(f"{WIDTH + 100}x{HEIGHT * 2.5}")
            case _:
                app.geometry(f"{WIDTH}x{HEIGHT}")
        pass

    tabview = CTkTabview(
        app,
        command=tabview_callback,
        corner_radius=5,
    )
    tabview.pack(
        padx=10,
        pady=10,
        fill="both",
        expand=True,
    )

    analysis_frame = tabview.add("Analysis")
    visuals_frame = tabview.add("Visuals")
    logreg_frame = tabview.add("Logreg")
    bonuses_frame = tabview.add("Bonus")
    tabview.set("Analysis")
    create_analysis_tab(tabview)
    create_visuals_tab(tabview)
    create_logreg_tab(tabview)
    create_bonus_tab(tabview)


def create_analysis_tab(tabs: CTkTabview):
    own_table = CTkTable(
        master=tabs.tab("Analysis"),
        row=11 + 1,
        column=5 + 1,
        border_color="#404040",
        border_width=3,
        corner_radius=CORNER_RADIUS // 2,
    )
    panda_table = CTkTable(
        master=tabs.tab("Analysis"),
        row=8 + 1,
        column=5 + 1,
        border_color="#404040",
        border_width=3,
        corner_radius=CORNER_RADIUS // 2,
    )

    def describe_dataset():
        df = read_csv("../datasets/dataset_train.csv")  \
              .drop("Index", axis=1)              \
              .select_dtypes(include="number")
        with redirect_stdout(open(devnull, "w")):
            own_describe, pd_describe = describe(df)
        cols = ["Astronomy", "Herbology", "Divination", "Potions", "Charms"]
        own_describe = own_describe[cols].round(2)
        own_values = [[""] + cols]
        for idx, row in own_describe.iterrows():
            own_values.append([idx] + row.values.tolist())
        pd_describe = pd_describe[cols].round(2)
        pd_values = [[""] + cols]
        for idx, row in pd_describe.iterrows():
            pd_values.append([idx] + row.values.tolist())
        own_table.configure(
            values=own_values
        )
        panda_table.configure(
            values=pd_values
        )
        own_table.pack(
            side="top",
            expand=False,
            padx=20,
            pady=10,
            anchor="n"
        )
        panda_table.pack(
            side="top",
            expand=False,
            padx=20,
            pady=10,
            anchor="n"
        )
        global HAS_DESCRIBED
        HAS_DESCRIBED = True
        app.geometry(f"{WIDTH}x800")

    tab_button_builder = ButtonBuilder(tabs.tab("Analysis"))
    tab_button_builder.new()            \
        .text("Describe")               \
        .size(60, 30)                   \
        .fg_color("#a1a61f")            \
        .hover(True, "#6d7014")         \
        .command(describe_dataset)      \
        .corner_radius(CORNER_RADIUS)   \
        .pack(
            side="top",
            anchor="n",
            pady=10,
        )


def create_visuals_tab(tabs: CTkTabview):
    wrapper = CTkFrame(tabs.tab("Visuals"), fg_color="transparent")
    wrapper.pack(fill="x", pady=20)

    button_row = CTkFrame(wrapper, fg_color="transparent")
    button_row.pack(pady=10)
    grid_frame = CTkFrame(wrapper, fg_color="transparent")
    checkbox = CTkCheckBox(grid_frame, text="Replace")
    def checkbox_callback():
        checkbox.configure(text="Replace")
        if checkbox.get():
            checkbox.configure(text="Append")
    checkbox.configure(command=checkbox_callback)

    #
    entry = CTkEntry(wrapper, placeholder_text="Plot to show...")
    entry.configure(corner_radius=5, font=("font", 13), width=250)
    entry.pack(padx=10, pady=10, expand=1, fill="both")
    grid_frame.pack(pady=10)

    def show_plot(plot: str):
        script_name = sub('([a-z])([A-Z])', r'\1 \2', plot)
        script_name = script_name.lower().replace(' ', '_')
        script_name += ".py"
        script_name = "/visualization/" + script_name

        @threaded
        def generate_callback():
            parameters = entry.get().split(', ')
            if len(entry.get()) != 0:
                execute_script(script_name, parameters)
            else:
                execute_script(script_name)


        return generate_callback


    def add_top_button(text):
        tab_button_builder = ButtonBuilder(button_row)
        tab_button_builder.new()        \
            .text(text)                 \
            .size(110, 35)              \
            .fg_color("#a1a61f")        \
            .hover(True, "#6d7014")     \
            .corner_radius(5)           \
            .command(show_plot(text))   \
            .pack(
                side="left",
                padx=10
            )


    add_top_button("Histogram")
    add_top_button("Scatter Plot")
    add_top_button("Pair Plot")

    def generate_button_callback(name: str):
        def callback():
            start = 0
            if not checkbox.get(): # Replace
                entry.delete(0, len(entry.get()))
            else: # Append
                start = len(entry.get()) + 2
                if start != 2:
                    entry.insert(start - 2, ", ")
            entry.insert(start, name)


        return callback

    columns = 3
    for i, course in enumerate(COURSES):
        r = i // columns
        c = i % columns

        tab_button_builder = ButtonBuilder(grid_frame)
        tab_button_builder.new()                        \
            .text(course)                               \
            .size(120, 35)                              \
            .fg_color("#1f7ba6")                        \
            .hover(True, "#145370")                     \
            .corner_radius(5)                           \
            .command(generate_button_callback(course))  \
            .grid(
                row=r,
                column=c,
                padx=10,
                pady=10,
            )
    checkbox.grid(row=4, column=2)

    for col in range(columns):
        grid_frame.grid_columnconfigure(col, weight=1)


def create_logreg_tab(tabs: CTkTabview):
    wrapper = CTkFrame(tabs.tab("Logreg"), fg_color="transparent")
    wrapper.pack(fill="x", pady=20)

    button_row = CTkFrame(wrapper, fg_color="transparent")
    button_row.pack(pady=10)

    def generate_logreg_callback(action: str):
        dataset = "../datasets/dataset_train.csv"
        if action.lower() == "predict":
            dataset = "../datasets/dataset_test.csv"
        def callback():
            script = "/regression/logreg_" + action.lower() + ".py"
            execute_script(script, [dataset])


        return callback


    def add_top_button(text):
        tab_button_builder = ButtonBuilder(button_row)
        tab_button_builder.new()                        \
            .text(text)                                 \
            .size(110, 35)                              \
            .fg_color("#a1a61f")                        \
            .hover(True, "#6d7014")                     \
            .corner_radius(5)                           \
            .command(generate_logreg_callback(text))    \
            .pack(
                side="left",
                padx=10
            )

    add_top_button("Train")
    add_top_button("Predict")


def create_bonus_tab(tabs: CTkTabview):
    wrapper = CTkFrame(tabs.tab("Bonus"), fg_color="transparent")
    wrapper.pack(fill="x", pady=20)

    button_row = CTkFrame(wrapper, fg_color="transparent")
    button_row.pack(pady=10)

    def bonus_logreg_train():
        print("I did not crash, just really slow")
        dataset = "../datasets/dataset_train.csv"
        script = "/bonus/logreg_train_bonus.py"
        execute_script(script, [dataset])


    def logreg_predict():
        dataset = "../datasets/dataset_test.csv"
        script = "/regression/logreg_predict.py"
        execute_script(script, [dataset])


    def generate_bonus_callback(script: str):
        script = "/bonus/" + script + ".py"
        @threaded
        def callback():
            execute_script(script)


        return callback

    def add_top_button(text, callback):
        tab_button_builder = ButtonBuilder(button_row)
        tab_button_builder.new()                        \
            .text(text)                                 \
            .size(110, 35)                              \
            .fg_color("#a1a61f")                        \
            .hover(True, "#6d7014")                     \
            .corner_radius(5)                           \
            .command(callback)                          \
            .pack(
                side="left",
                padx=10
            )

    add_top_button("Bonus Train", bonus_logreg_train)
    add_top_button("Predict", logreg_predict)
    add_top_button("Accuracy", generate_bonus_callback("accuracy_visualizer"))
    add_top_button("Boundaries", generate_bonus_callback("boundaries_line_visualizer"))


def init_app():
    app.geometry(f"{WIDTH}x{HEIGHT}")
    create_exit_button()
    create_tabs()


def main():
    try:
        init_app()
        app.mainloop()

    except Exception as err:
        print_error(f"Unexpected error: {err}")


if __name__ == "__main__":
    main()