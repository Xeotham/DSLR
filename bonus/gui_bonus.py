#!../.venv/bin/python
import sys
sys.path.insert(1, "..")
sys.path.insert(2, ".")

from re import search, split, sub, match
from io import StringIO
from subprocess import run
from builders.button_builder import ButtonBuilder
from customtkinter import *
from dslr_lib.errors import print_error
from pandas import read_csv

app = CTk()
button_builder = ButtonBuilder(app)


def execute_script(path: str, args: str, stdout: str):
    absolute_path = __file__.replace("/bonus/./gui_bonus.py", "")
    absolute_path += path
    with open(stdout, "w") as f:
        run([absolute_path, args], stdout=f)


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
    tabview = CTkTabview(app)
    tabview.pack(
        padx=10,
        pady=10,
        fill="both",
        expand=1
    )
    tabview.add("Analysis")
    tabview.add("Visuals")
    tabview.add("Logreg")
    tabview.add("Bonuses")
    tabview.set("Analysis")
    create_analysis_tab(tabview)


def create_analysis_tab(tabs: CTkTabview):
    def describe_dataset():
        execute_script(
            "/analysis/describe.py",
            "../datasets/dataset_train.csv",
            ".analysis"
        )
        def extract_tables():
            with open(".analysis") as f:
                buffer = f.read()
            matched = search('\n', buffer)
            colnum = int(buffer[search(" x ", buffer).start() + 3:search(" x ", buffer).start() + 6])
            
            print(colnum)
            buffer = buffer[matched.start() + 1::]
            print(buffer)
            # return table1, table2
        describe_output, pandas_output = extract_tables()
        # print(describe_output)
        # print(pandas_output)




    tab_button_builder = ButtonBuilder(tabs.tab("Analysis"))
    tab_button_builder.new()            \
        .text("Describe")               \
        .size(60, 30)                   \
        .fg_color("#a1a61f")            \
        .hover(True, "#6d7014")         \
        .command(describe_dataset)      \
        .pack(
            side="top",
            padx=5,
            pady=5,
        )

def init_app():
    app.geometry("800x600")
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