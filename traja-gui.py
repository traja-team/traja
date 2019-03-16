from os.path import basename
import sys

import matplotlib

matplotlib.use("Qt5Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtGui, QtWidgets

import traja

style.use("ggplot")


class PlottingWidget(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # super(PrettyWidget, self).__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(600, 300, 1000, 600)
        self.center()
        self.setWindowTitle("Plot Trajectory")
        # Grid Layout
        grid = QtWidgets.QGridLayout()
        widget = QtWidgets.QWidget(self)
        self.setCentralWidget(widget)
        widget.setLayout(grid)

        # Import CSV Button
        btn1 = QtWidgets.QPushButton("Import CSV", self)
        btn1.resize(btn1.sizeHint())
        btn1.clicked.connect(self.getCSV)
        grid.addWidget(btn1, 1, 0)

        # Canvas and Toolbar
        self.figure = plt.figure(figsize=(15, 5))
        self.canvas = FigureCanvas(self.figure)
        grid.addWidget(self.canvas, 2, 0, 1, 2)

        # DropDown mean / comboBox
        self.df = pd.DataFrame()
        self.columns = []
        self.plot_list = []

        self.comboBox = QtWidgets.QComboBox(self)
        self.comboBox.addItems(self.columns)
        grid.addWidget(self.comboBox, 0, 0)

        self.comboBox2 = QtWidgets.QComboBox(self)
        self.comboBox2.addItems(self.plot_list)
        grid.addWidget(self.comboBox2, 0, 1)

        # Plot Button
        btn2 = QtWidgets.QPushButton("Plot", self)
        btn2.resize(btn2.sizeHint())
        btn2.clicked.connect(self.plot)
        grid.addWidget(btn2, 1, 1)

        self.show()

    def getCSV(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", "/home")
        if filepath != "":
            self.filepath = filepath
            self.df = traja.read_file(
                str(filepath),
                index_col="time_stamps_vec",
                parse_dates=["time_stamps_vec"],
            )
            self.columns = self.df.columns.tolist()
            self.plot_list = ["Actogram", "Polar", "Trajectory"]
            self.comboBox.addItems(self.columns)
            self.comboBox2.addItems(self.plot_list)

    def plot(self):
        plt.clf()

        plot_kind = self.comboBox2.currentText()
        projection = "polar" if plot_kind in ["Polar"] else "rectilinear"

        ax = self.figure.add_subplot(111, projection=projection)

        title = f"{basename(self.filepath)}"
        # TODO: Move mapping to separate method
        if plot_kind == "Actogram":
            displacement = traja.trajectory.calc_displacement(self.df)
            traja.plotting.plot_actogram(displacement, ax=ax, interactive=False)
        elif plot_kind == "Trajectory":
            traja.plotting.plot(self.df, ax=ax, interactive=False)
        elif plot_kind == "Polar":
            traja.plotting.polar_bar(self.df, ax=ax, title=title, interactive=False)
        plt.tight_layout()
        self.canvas.draw()

    def center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = PlottingWidget()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
