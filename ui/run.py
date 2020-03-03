import sys

from PyQt5 import QtCore, QtGui, QtWidgets

from main import Main_Dialog

# from details import Details_Dialog as Main_Dialog

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = Main_Dialog()
    ui.show()
    sys.exit(app.exec_())
