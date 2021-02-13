import os
from os.path import basename
from functools import partial
import sys

import matplotlib

matplotlib.use("Qt5Agg")
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()

import matplotlib.style as style
import matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QProgressBar, QMenu, QAction, QStatusBar

import traja

CUR_STYLE = "fast"
style.use(CUR_STYLE)
TIME_WINDOW = "30s"


class QtFileLoader(QObject):
    finished = pyqtSignal()
    progressMaximum = pyqtSignal(int)
    completed = pyqtSignal(list)
    intReady = pyqtSignal(int)

    def __init__(self, filepath):
        super(QtFileLoader, self).__init__()
        self.filepath = filepath

    @pyqtSlot()
    def read_in_chunks(self):
        """ load dataset in parts and update the progess par """
        chunksize = 10 ** 3
        lines_number = sum(1 for line in open(self.filepath))
        self.progressMaximum.emit(lines_number // chunksize)
        dfList = []

        # self.df = traja.read_file(
        #     str(filepath),
        #     index_col="time_stamps_vec",
        #     parse_dates=["time_stamps_vec"],
        # )

        TextFileReader = pd.read_csv(
            self.filepath,
            index_col="time_stamps_vec",
            parse_dates=["time_stamps_vec"],
            chunksize=chunksize,
        )
        for idx, df in enumerate(TextFileReader):
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S:%f")
            dfList.append(df)
            self.intReady.emit(idx)
        self.completed.emit(dfList)
        self.finished.emit()


class PlottingWidget(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # super(PrettyWidget, self).__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(600, 300, 1000, 600)
        self.center()
        self.setWindowTitle("Plot Trajectory")

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")

        saveAction = QAction("Save as...")
        saveAction.setShortcut("Ctrl+S")
        saveAction.setStatusTip("Save plot to file")
        saveAction.setMenuRole(QAction.NoRole)
        saveAction.triggered.connect(self.file_save)
        fileMenu.addAction(saveAction)

        exitAction = QAction("&Exit", self)
        exitAction.setShortcut("Ctrl+Q")
        exitAction.setStatusTip("Exit Application")
        exitAction.setMenuRole(QAction.NoRole)
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

        settingsMenu = mainMenu.addMenu("Settings")
        self.setStyleMenu = QMenu("Set Style", self)
        settingsMenu.addMenu(self.setStyleMenu)
        for style_name in ["default", "fast", "ggplot", "grayscale", "seaborn"]:
            styleAction = QAction(style_name, self, checkable=True)
            if style_name is CUR_STYLE:
                styleAction.setChecked(True)
            styleAction.triggered.connect(partial(self.set_style, style_name))
            self.setStyleMenu.addAction(styleAction)
        self.setTimeWindowMenu = QMenu("Set Time Window", self)
        settingsMenu.addMenu(self.setTimeWindowMenu)
        for window_str in ["None", "s", "30s", "H", "D"]:
            windowAction = QAction(window_str, self, checkable=True)
            if window_str is TIME_WINDOW:
                windowAction.setChecked(True)
            windowAction.triggered.connect(partial(self.set_time_window, window_str))
            self.setTimeWindowMenu.addAction(windowAction)

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
        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self.popup)
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

        # Progress bar
        self.progress = QProgressBar(self)
        # self.progress.setRange(0, 1)
        grid.addWidget(self.progress, 3, 0, 1, 2)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.show()

    def set_style(self, style_name: str):
        global CUR_STYLE
        self.statusBar.showMessage(f"Style set to {style_name}")
        actions = self.setStyleMenu.actions()
        CUR_STYLE = style_name
        for action in actions:
            if action.text() == CUR_STYLE:
                # print(f"✓ {CUR_STYLE}")
                action.setChecked(True)
            else:
                action.setChecked(False)
        print(f"Style set to {CUR_STYLE}")

    def popup(self, pos):
        menu = QMenu()
        saveAction = menu.addAction("Save...")
        action = menu.exec_(self.canvas.viewport().mapToGlobal(pos))
        if action == saveAction:
            self.file_save()

    def file_save(self, target="figure"):
        name = QtGui.QFileDialog.getSaveFileName(self, "Save File")
        if target == "figure":
            self.figure.savefig(name)

    def update_progress_bar(self, i: int):
        self.progress.setValue(i)
        max = self.progress.maximum()
        self.statusBar.showMessage(f"Loading ... {100*i/max:.0f}%")

    def set_progress_bar_max(self, max: int):
        self.progress.setMaximum(max)

    def clear_progress_bar(self):
        self.progress.hide()
        self.statusBar.showMessage("Completed.")

    def getCSV(self):
        self.statusBar.showMessage("Loading CSV...")
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open CSV", (QtCore.QDir.homePath()), "CSV (*.csv *.tsv)"
        )

        if filepath != "":
            self.filepath = filepath
            self.loaderThread = QThread()
            self.loaderWorker = QtFileLoader(filepath)
            self.loaderWorker.moveToThread(self.loaderThread)
            self.loaderThread.started.connect(self.loaderWorker.read_in_chunks)
            self.loaderWorker.intReady.connect(self.update_progress_bar)
            self.loaderWorker.progressMaximum.connect(self.set_progress_bar_max)
            # self.loaderWorker.read_in_chunks.connect(self.df)
            self.loaderWorker.completed.connect(self.list_to_df)
            self.loaderWorker.completed.connect(self.clear_progress_bar)
            self.loaderThread.finished.connect(self.loaderThread.quit)
            self.loaderThread.start()

    @pyqtSlot(list)
    def list_to_df(self, dfs: list):
        df = pd.concat(dfs)
        self.df = df
        self.columns = self.df.columns.tolist()
        self.plot_list = ["Actogram", "Polar Bar", "Polar Histogram", "Trajectory"]
        self.comboBox.clear()
        self.comboBox.addItems(self.columns)
        self.comboBox2.clear()
        self.comboBox2.addItems(self.plot_list)
        self.statusBar.clearMessage()

    def mousePressEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.RightButton:

            print("Right Button Clicked")

    def load_project_structure(self, startpath, tree):
        """
        Load Project structure tree
        :param startpath:
        :param tree:
        :return:
        """
        from PyQt5.QtWidgets import QTreeWidgetItem
        from PyQt5.QtGui import QIcon

        for element in os.listdir(startpath):
            path_info = startpath + "/" + element
            parent_itm = QTreeWidgetItem(tree, [os.path.basename(element)])
            if os.path.isdir(path_info):
                self.load_project_structure(path_info, parent_itm)
                parent_itm.setIcon(0, QIcon("assets/folder.ico"))
            else:
                parent_itm.setIcon(0, QIcon("assets/file.ico"))

    def set_time_window(self, window: str):
        global TIME_WINDOW
        TIME_WINDOW = window
        self.statusBar.showMessage(f"Time window set to {window}")
        actions = self.setTimeWindowMenu.actions()
        for action in actions:
            if action.text() == TIME_WINDOW:
                action.setChecked(True)
            else:
                action.setChecked(False)
        print(f"Time window set to {window}")

    def plot(self):
        plt.clf()

        plot_kind = self.comboBox2.currentText()
        self.statusBar.showMessage(f"Plotting {plot_kind}")
        projection = (
            "polar" if plot_kind in ["Polar Bar", "Polar Histogram"] else "rectilinear"
        )

        ax = self.figure.add_subplot(111, projection=projection)

        title = f"{basename(self.filepath)}"

        # TODO: Move mapping to separate method
        if plot_kind == "Actogram":
            displacement = traja.trajectory.calc_displacement(self.df)
            if TIME_WINDOW != "None":
                displacement = displacement.rolling(TIME_WINDOW).mean()
                # from pyqtgraph.Qt import QtGui, QtCore
            traja.plotting.plot_actogram(displacement, ax=ax, interactive=False)
        elif plot_kind == "Trajectory":
            traja.plotting.plot(self.df, ax=ax, interactive=False)
        elif plot_kind == "Quiver":
            traja.plotting.plot_quiver(self.df, ax=ax, interactive=False)
        elif plot_kind == "Polar Bar":
            traja.plotting.polar_bar(self.df, ax=ax, title=title, interactive=False)
        elif plot_kind == "Polar Histogram":
            traja.plotting.polar_bar(
                self.df, ax=ax, title=title, overlap=False, interactive=False
            )
        plt.tight_layout()
        self.canvas.draw()
        self.statusBar.clearMessage()

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
