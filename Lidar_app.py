import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFrame, QHBoxLayout, \
    QWidget, QStackedWidget, QComboBox, QLineEdit
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QPropertyAnimation, QMutex
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import os
from ICP_function import icp_algorithm
from Lidar_classes import STL27L, A2M8
import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class WaveWorker(QObject):
    queue_W = pyqtSignal(list)
    finished_W = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.Wave_lidar = STL27L()

    # noinspection PyUnresolvedReferences
    def do_work(self):
        self.is_running = True
        if self.Wave_lidar.is_active is True:
            while self.is_running:
                data_wave = self.Wave_lidar.make_full_scan()
                self.queue_W.emit(data_wave)

            self.Wave_lidar.deactivate()
        self.finished_W.emit()

    def stop_work(self):
        self.is_running = False

    def check_lidar_status(self):
        if self.Wave_lidar.is_active is True:
            return True
        return False

    def save_lidar_scan(self):
        self.Wave_lidar.save_scan_to_csv()


class SlamWorker(QObject):
    queue_S = pyqtSignal(list)
    finished_S = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.Slam_lidar = A2M8()
        self.previous_angle = 0
        self.previous_start_angle = 0

    # noinspection PyUnresolvedReferences
    def do_work(self):
        self.is_running = True
        if self.Slam_lidar.is_active is True:
            self.Slam_lidar.reset()
            self.Slam_lidar.run()
            while self.is_running:
                data_slam, self.previous_angle, self.previous_start_angle = self.Slam_lidar.make_full_scan(
                    self.previous_angle, self.previous_start_angle)
                self.queue_S.emit(data_slam)
            self.Slam_lidar.deactivate()
        self.finished_S.emit()

    def stop_work(self):
        self.is_running = False

    def check_lidar_status(self):
        if self.Slam_lidar.is_active is True:
            return True
        return False

    def save_lidar_scan(self):
        self.Slam_lidar.save_scan_to_csv()

# noinspection PyUnresolvedReferences,PyTypeChecker
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.CustomizeWindowHint)
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        loadUi('UI/UI_lidar_app.ui', self)
        self.central_widget = self.findChild(QWidget, 'centralwidget')

        # Header init
        self.header_widget = self.findChild(QStackedWidget, 'header')
        self.header_exit = self.findChild(QWidget, 'exit')
        self.header_back = self.findChild(QWidget, 'back')
        # Header buttons
        self.back_button = self.findChild(QPushButton, 'BackButton')
        self.exit_button = self.findChild(QPushButton, 'ExitButton')
        # Header actions
        self.header_widget.setCurrentWidget(self.header_exit)
        self.exit_button.clicked.connect(self.exit_application)
        self.back_button.clicked.connect(self.main_page)

        # Main Page init
        self.mb_widget = self.findChild(QStackedWidget, 'mainbody')
        self.mb_mainPage = self.findChild(QWidget, 'MB_mainPage')
        self.mb_page1 = self.findChild(QWidget, 'MB_Page1')
        self.mb_page2 = self.findChild(QWidget, 'MB_Page2')
        self.mb_page3 = self.findChild(QWidget, 'MB_Page3')
        self.mb_page4 = self.findChild(QWidget, 'MB_Page4')
        self.mb_page5 = self.findChild(QWidget, 'MB_Page5')
        # Main Page buttons
        self.case1_button = self.findChild(QPushButton, 'Case1Button')
        self.case2_button = self.findChild(QPushButton, 'Case2Button')
        self.case3_button = self.findChild(QPushButton, 'Case3Button')
        self.case4_button = self.findChild(QPushButton, 'Case4Button')
        self.case5_button = self.findChild(QPushButton, 'Case5Button')
        # Main Page actions
        self.mb_widget.setCurrentWidget(self.mb_mainPage)
        self.case1_button.clicked.connect(self.show_page1)
        self.case2_button.clicked.connect(self.show_page2)
        self.case3_button.clicked.connect(self.show_page3)
        self.case4_button.clicked.connect(self.show_page4)
        self.case5_button.clicked.connect(self.show_page5)

        # Case buttons animation init
        start_width_1 = self.case1_button.geometry().width()
        self.animation_case1 = QPropertyAnimation(self.case1_button, b"minimumWidth")
        self.animation_case1.setDuration(300)
        self.animation_case1.setStartValue(start_width_1)
        self.animation_case1.setEndValue(start_width_1*1.1)

        start_width_2 = self.case2_button.geometry().width()
        self.animation_case2 = QPropertyAnimation(self.case2_button, b"minimumWidth")
        self.animation_case2.setDuration(300)
        self.animation_case2.setStartValue(start_width_2)
        self.animation_case2.setEndValue(start_width_2 * 1.1)

        start_width_3 = self.case3_button.geometry().width()
        self.animation_case3 = QPropertyAnimation(self.case3_button, b"minimumWidth")
        self.animation_case3.setDuration(300)
        self.animation_case3.setStartValue(start_width_3)
        self.animation_case3.setEndValue(start_width_3 * 1.1)

        start_width_4 = self.case4_button.geometry().width()
        self.animation_case4 = QPropertyAnimation(self.case4_button, b"minimumWidth")
        self.animation_case4.setDuration(300)
        self.animation_case4.setStartValue(start_width_4)
        self.animation_case4.setEndValue(start_width_4 * 1.1)

        start_width_5 = self.case5_button.geometry().width()
        self.animation_case5 = QPropertyAnimation(self.case5_button, b"minimumWidth")
        self.animation_case5.setDuration(300)
        self.animation_case5.setStartValue(start_width_5)
        self.animation_case5.setEndValue(start_width_5 * 1.1)

        # Animation work
        self.case1_button.enterEvent = lambda event: self.expand_button(event, self.animation_case1)
        self.case1_button.leaveEvent = lambda event: self.restore_button(event, self.animation_case1)

        self.case2_button.enterEvent = lambda event: self.expand_button(event, self.animation_case2)
        self.case2_button.leaveEvent = lambda event: self.restore_button(event, self.animation_case2)

        self.case3_button.enterEvent = lambda event: self.expand_button(event, self.animation_case3)
        self.case3_button.leaveEvent = lambda event: self.restore_button(event, self.animation_case3)

        self.case4_button.enterEvent = lambda event: self.expand_button(event, self.animation_case4)
        self.case4_button.leaveEvent = lambda event: self.restore_button(event, self.animation_case4)

        self.case5_button.enterEvent = lambda event: self.expand_button(event, self.animation_case5)
        self.case5_button.leaveEvent = lambda event: self.restore_button(event, self.animation_case5)

        # Page 1 layout and canvas
        self.page1_plt = self.findChild(QFrame, 'frame_page1')
        self.hLayout1 = QHBoxLayout(self.page1_plt)
        self.fig1, self.ax1 = plt.subplots()
        self.canvas1 = FigureCanvas(self.fig1)
        self.hLayout1.addWidget(self.canvas1)
        self.plot1 = None
        self.plot2 = None
        # Page 1 buttons
        self.start_button = self.findChild(QPushButton, 'StartButton')
        self.stop_button = self.findChild(QPushButton, 'StopButton')
        self.saveScan_button = self.findChild(QPushButton, 'SaveScanButton')
        # Page 1 line edit init
        self.radius_value = self.findChild(QLineEdit, 'RadiusValue')
        # Page 1 threads init
        self.Wave_thread = None
        self.Wave_worker = None
        self.Slam_thread = None
        self.Slam_worker = None
        # Page 1 actions
        self.start_button.clicked.connect(self.start_worker)
        self.stop_button.clicked.connect(self.stop_worker)

        # Page 2 layout and canvas
        self.page2_plt = self.findChild(QFrame, 'frame_page2')
        self.hLayout2 = QHBoxLayout(self.page2_plt)
        self.fig2, self.ax2 = plt.subplots()
        self.canvas2 = FigureCanvas(self.fig2)
        self.hLayout2.addWidget(self.canvas2)
        """self.page2_bar = self.findChild(QFrame, 'frame')
        self.bLayout2 = QHBoxLayout(self.page2_bar)
        toolbar2 = NavigationToolbar(self.canvas2, self)
        toolbar2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.bLayout2.addWidget(toolbar2)
        self.change_toolbar_icon_color(toolbar2, "white")"""
        # Page 2 combo box init
        self.choose_scan1 = self.findChild(QComboBox, 'Scan1Box')
        self.choose_scan2 = self.findChild(QComboBox, 'Scan2Box')
        # Page 2 buttons
        self.runICP_button = self.findChild(QPushButton, 'MergeButton')
        # Page 3 line edit init
        self.rmseICP_val = self.findChild(QLineEdit, 'RmseIcpValue')
        # Page 2 actions
        self.runICP_button.clicked.connect(self.icp_result)

        # Page 3 layout and canvas
        self.page3_plt = self.findChild(QFrame, 'frame_page3')
        self.hLayout3 = QHBoxLayout(self.page3_plt)
        self.fig3, self.ax3 = plt.subplots()
        self.canvas3 = FigureCanvas(self.fig3)
        self.hLayout3.addWidget(self.canvas3)
        self.canvas3.mpl_connect('scroll_event', self.zoom)
        self.canvas3.mpl_connect('button_press_event', self.on_press)
        # Page 3 combo box init
        self.choose_scan3 = self.findChild(QComboBox, 'Scan3Box')
        # Page 3 buttons
        self.sliceScan_button = self.findChild(QPushButton, 'SaveChopButton')
        self.refresh_button = self.findChild(QPushButton, 'RefreshButton')
        # Page 3 line edit init
        self.slice_name = self.findChild(QLineEdit, 'ChopName')
        # Page 3 actions
        self.refresh_button.clicked.connect(self.page3_plot_refresh)
        self.selected_points = np.empty(0)
        self.sliceScan_button.clicked.connect(self.save_slice_to_csv)
        # self.slice_name.textChanged.connect(self.check_slice)

        # Page 4 layout and canvas
        self.page4_plt = self.findChild(QFrame, 'frame_page4')
        self.hLayout4 = QHBoxLayout(self.page4_plt)
        self.fig4, self.ax4 = plt.subplots()
        self.canvas4 = FigureCanvas(self.fig4)
        self.hLayout4.addWidget(self.canvas4)
        # Page 4 combo box init
        self.choose_scan4 = self.findChild(QComboBox, 'Scan4Box')
        # Page 4 buttons
        self.runRMSE_button = self.findChild(QPushButton, 'CalcButton')
        # Page 4 line edit init
        self.rmse_val = self.findChild(QLineEdit, 'RmseValue')
        self.pear_val = self.findChild(QLineEdit, 'PearValue')
        self.len_val = self.findChild(QLineEdit, 'LenValue')
        self.dist_val = self.findChild(QLineEdit, 'DistanceValue')
        # Page 4 actions
        self.runRMSE_button.clicked.connect(self.rmse_result)

        # Page 5 layout and canvas
        self.page5_plt = self.findChild(QFrame, 'frame_page5')
        self.hLayout5 = QHBoxLayout(self.page5_plt)
        self.fig5, self.ax5 = plt.subplots()
        self.canvas5 = FigureCanvas(self.fig5)
        self.hLayout5.addWidget(self.canvas5)
        # Page 5 combo box init
        self.choose_scan5 = self.findChild(QComboBox, 'AngleBox')
        # Page 5 buttons
        self.calcANG_button = self.findChild(QPushButton, 'CalcAngButton')
        # Page 5 line edit init
        self.ang_val = self.findChild(QLineEdit, 'AngleValue')
        # Page 5 actions
        self.calcANG_button.clicked.connect(self.ang_result)

        # Mutex
        self.mutex = QMutex()

    def expand_button(self, event, animation):
        if animation.state() == QPropertyAnimation.Running:
            animation.stop()
        animation.setDirection(QPropertyAnimation.Forward)
        animation.start()

    def restore_button(self, event, animation):
        if animation.state() == QPropertyAnimation.Running:
            animation.stop()
        animation.setDirection(QPropertyAnimation.Backward)
        animation.start()

    def start_worker(self):
        value = self.radius_value.text()
        if value.isdigit() and not self.stop_button.isEnabled() and int(value) < 100000:
            radius = int(value)
            self.ax1.clear()
            self.ax1.set_xlim(-1 * radius, radius)
            self.ax1.set_ylim(-1 * radius, radius)
            self.ax1.grid(True, color='gray')  # Dodaj siatkę do wykresu
            self.plot1, = self.ax1.plot([], [], 'bo', markersize=1, label='A2M8')
            self.plot2, = self.ax1.plot([], [], 'ro', markersize=1, label='STL27L')
            self.ax1.legend()

            self.Slam_worker = SlamWorker()
            if self.Slam_worker.check_lidar_status() is True:
                self.Slam_thread = QThread()
                self.Slam_worker.finished_S.connect(self.on_finished)
                self.Slam_worker.queue_S.connect(self.process_slam_data)
                self.Slam_worker.moveToThread(self.Slam_thread)
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                if not self.Slam_thread.isRunning():
                    self.Slam_thread.start()
                    self.Slam_thread.started.connect(self.Slam_worker.do_work)

            self.Wave_worker = WaveWorker()
            if self.Wave_worker.check_lidar_status() is True:
                self.Wave_thread = QThread()
                self.Wave_worker.finished_W.connect(self.on_finished)
                self.Wave_worker.queue_W.connect(self.process_wave_data)
                self.Wave_worker.moveToThread(self.Wave_thread)
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                if not self.Wave_thread.isRunning():
                    self.Wave_thread.start()
                    self.Wave_thread.started.connect(self.Wave_worker.do_work)

            if self.Slam_worker.check_lidar_status() is True or self.Wave_worker.check_lidar_status() is True:
                self.saveScan_button.setEnabled(True)
                self.saveScan_button.clicked.connect(self.init_save_scan)
        else:
            self.radius_value.setText("")

    def init_save_scan(self):
        if self.Slam_worker.check_lidar_status() is True:
            self.Slam_worker.save_lidar_scan()

        if self.Wave_worker.check_lidar_status() is True:
            self.Wave_worker.save_lidar_scan()

    def process_slam_data(self, data_slam):
        """self.mutex.lock()
        try:
            self.update_plot(data_slam[0], data_slam[1], [], [])
        finally:
            self.mutex.unlock()"""
        self.update_plot(data_slam[0], data_slam[1], [], [])

    def process_wave_data(self, data_wave):
        self.update_plot([], [], data_wave[0], data_wave[1])
    

    def update_plot(self, x1, y1, x2, y2):
        if x1 and y1:
            self.plot1.set_xdata(x1)
            self.plot1.set_ydata(y1)
        if x2 and y2:
            self.plot2.set_xdata(x2)
            self.plot2.set_ydata(y2)

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.canvas1.draw()

    def on_finished(self):
        if self.Slam_worker.check_lidar_status() is True:
            self.Slam_thread.quit()
            self.Slam_thread.wait()
            print("Slamtec worker finished.")

        if self.Wave_worker.check_lidar_status() is True:
            self.Wave_thread.quit()
            self.Wave_thread.wait()
            print("Waveshare worker finished.")

    def stop_worker(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.saveScan_button.setEnabled(False)

        if self.Slam_worker.check_lidar_status() is True:
            self.Slam_worker.stop_work()
        if self.Wave_worker.check_lidar_status() is True:
            self.Wave_worker.stop_work()

    def exit_application(self):
        QApplication.quit()

    def show_page1(self):
        self.mb_widget.setCurrentWidget(self.mb_page1)
        self.header_widget.setCurrentWidget(self.header_back)

    def show_page2(self):
        self.mb_widget.setCurrentWidget(self.mb_page2)
        self.header_widget.setCurrentWidget(self.header_back)
        self.choose_scan1.clear()
        self.choose_scan2.clear()
        file_names = os.listdir("Slices/")
        for file_name in file_names:
            self.choose_scan1.addItem(file_name)
            self.choose_scan2.addItem(file_name)

    def show_page3(self):
        self.mb_widget.setCurrentWidget(self.mb_page3)
        self.header_widget.setCurrentWidget(self.header_back)
        self.choose_scan3.clear()
        file_names = os.listdir("Scans/")
        for file_name in file_names:
            self.choose_scan3.addItem(file_name)

    def show_page4(self):
        self.mb_widget.setCurrentWidget(self.mb_page4)
        self.header_widget.setCurrentWidget(self.header_back)
        self.choose_scan4.clear()
        file_names = os.listdir("Slices/")
        for file_name in file_names:
            self.choose_scan4.addItem(file_name)

    def show_page5(self):
        self.mb_widget.setCurrentWidget(self.mb_page5)
        self.header_widget.setCurrentWidget(self.header_back)
        self.choose_scan5.clear()
        file_names = os.listdir("Slices/")
        for file_name in file_names:
            self.choose_scan5.addItem(file_name)

    def main_page(self):
        self.mb_widget.setCurrentWidget(self.mb_mainPage)
        self.header_widget.setCurrentWidget(self.header_exit)

    def icp_result(self):
        # Select current chosen 2 scans
        selected_file1 = self.choose_scan1.currentText()
        selected_file2 = self.choose_scan2.currentText()

        # Show merged plot
        self.ax2.clear()
        transformed_source_cloud, target_cloud, rmse_icp = icp_algorithm(selected_file1, selected_file2)

        self.rmseICP_val.setText(f"{rmse_icp:.4f}".replace('.', ','))

        self.ax2.set_xlabel('x [mm]')
        self.ax2.set_ylabel('y [mm]')
        self.ax2.scatter(transformed_source_cloud[:, 0], transformed_source_cloud[:, 1], color='blue', s=1)
        self.ax2.scatter(target_cloud[:, 0], target_cloud[:, 1], color='red', s=1)
        self.ax2.set_title('Merged point cloud')
        self.canvas2.draw()

    def on_press(self, event):
        if event.button == 1:
            self.x_start, self.y_start = event.xdata, event.ydata
            if self.x_start is None or self.y_start is None:
                pass
            else:
                self.rect = plt.Rectangle((self.x_start, self.y_start), 0, 0, alpha=0.3)
                self.ax3.add_patch(self.rect)
                self.fig3.canvas.draw_idle()
                self.fig3.canvas.mpl_connect('motion_notify_event', self.on_move)
                self.fig3.canvas.mpl_connect('button_release_event', self.on_release)
            # self.check_slice()

    def on_move(self, event):
        if event.button == 1:
            x_end, y_end = event.xdata, event.ydata
            self.rect.set_width(x_end - self.x_start)
            self.rect.set_height(y_end - self.y_start)
            self.rect.set_xy((self.x_start, self.y_start))
            self.fig3.canvas.draw()

    def on_release(self, event):
        if event.button == 1:
            x_end, y_end = event.xdata, event.ydata
            self.rect.set_width(x_end - self.x_start)
            self.rect.set_height(y_end - self.y_start)
            self.rect.set_color('r')
            self.selected_points = self.get_points_in_rect(self.x_start, y_end, x_end, self.y_start)
            self.ax3.figure.canvas.draw_idle()
            # self.check_slice()

    def check_slice(self):
        text = self.slice_name.text()

        if text and ' ' not in text and self.selected_points.any():
            self.sliceScan_button.setEnabled(True)
        else:
            self.sliceScan_button.setEnabled(False)

    def get_points_in_rect(self, x1, y1, x2, y2):
        mask = (self.selected_cloud[:, 0] >= x1) & (self.selected_cloud[:, 0] <= x2) & (
                self.selected_cloud[:, 1] >= y1) & (self.selected_cloud[:, 1] <= y2)
        return self.selected_cloud[mask]

    def zoom(self, event):
        # Skalowanie współczynników
        base_scale = 1.1

        # Sprawdzanie kierunku scrolla
        if event.button == 'up':
            scale_factor = base_scale
        elif event.button == 'down':
            scale_factor = 1 / base_scale
        else:
            return

        # Pobieranie aktualnych granic osi
        x_min, x_max = self.ax3.get_xlim()
        y_min, y_max = self.ax3.get_ylim()

        # Pobieranie współrzędnych wskaźnika myszy

        mouse_x, mouse_y = event.xdata, event.ydata

        if mouse_x is None or mouse_y is None:
            return  # Ignorowanie scrolla, gdy wskaźnik myszki jest poza wykresem

        # Obliczanie nowych granic osi
        new_x_min = mouse_x - (mouse_x - x_min) / scale_factor
        new_x_max = mouse_x + (x_max - mouse_x) / scale_factor
        new_y_min = mouse_y - (mouse_y - y_min) / scale_factor
        new_y_max = mouse_y + (y_max - mouse_y) / scale_factor

        # Ustawianie nowych granic osi
        self.ax3.set_xlim([new_x_min, new_x_max])
        self.ax3.set_ylim([new_y_min, new_y_max])

        # Odświeżanie wykresu
        self.canvas3.draw()

    def save_slice_to_csv(self):
        new_file = self.slice_name.text()
        if self.selected_points.size == 0:
            print("No points selected!")
            return
        if not new_file or ' ' in new_file:
            print("File name does not exist or contain spaces!")
            return
        if self.selected_points.size > 0 and len(new_file) > 0:
            filename = os.path.join('Slices', f"{new_file}.csv")
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                for point in self.selected_points:
                    writer.writerow(point)

    def page3_plot_refresh(self):
        selected_file = self.choose_scan3.currentText()
        self.selected_cloud = np.loadtxt(f'Scans/{selected_file}', delimiter=',')

        self.ax3.clear()
        self.ax3.set_xlabel('x [mm]')
        self.ax3.set_ylabel('y [mm]')
        self.ax3.scatter(self.selected_cloud[:, 0], self.selected_cloud[:, 1], color='blue', s=1)
        self.ax3.set_title('Select an area to crop')
        self.canvas3.draw()
        self.selected_points = np.empty(0)
        # self.check_slice()
        # self.canvas3.mpl_connect('button_press_event', self.on_press)

    def rmse_result(self):
        # Select current chosen slice
        selected_file = self.choose_scan4.currentText()
        selected_slice = np.loadtxt(f'Slices/{selected_file}', delimiter=',')

        x = selected_slice[:, 0]
        y = selected_slice[:, 1]
        # Linear regression
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        y_pred = model.predict(x.reshape(-1, 1))
        # RMS error
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        # Pearson correlation
        correlation_matrix = np.corrcoef(y, y_pred)
        correlation_coefficient = correlation_matrix[0, 1]
        # Length (largest distance vector)
        distances = np.sqrt((x[:, np.newaxis] - x[np.newaxis, :]) ** 2 + (y[:, np.newaxis] - y[np.newaxis, :]) ** 2)
        i, j = np.unravel_index(np.argmax(distances), distances.shape)
        longest_distance = distances[i, j]
        point1 = (x[i], y[i])
        point2 = (x[j], y[j])
        # Distance (distance from lidar to point cloud centroid)
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        distance_to_lidar = np.sqrt(centroid_x ** 2 + centroid_y ** 2)

        # Return calculated values
        self.rmse_val.setText(f"{rmse:.4f}".replace('.', ','))
        self.pear_val.setText(f"{correlation_coefficient:.4f}".replace('.', ','))
        self.len_val.setText(f"{longest_distance:.2f}".replace('.', ','))
        self.dist_val.setText(f"{distance_to_lidar:.2f}".replace('.', ','))

        # Show plot
        self.ax4.clear()
        self.ax4.set_xlabel('x [mm]')
        self.ax4.set_ylabel('y [mm]')
        self.ax4.scatter(x, y, label='Actual')
        self.ax4.scatter(x, y_pred, color='red', label='Predicted')
        self.ax4.plot([point1[0], point2[0]], [point1[1], point2[1]], color='green', linestyle='-',
                      linewidth=1, label=f'Length')
        self.ax4.legend()
        self.ax4.set_title(f'Linear regression and largest distance vector')
        self.canvas4.draw()

    def ang_result(self):
        # Select current chosen slice
        selected_file = self.choose_scan5.currentText()
        selected_slice = np.loadtxt(f'Slices/{selected_file}', delimiter=',')

        x = selected_slice[:, 0]
        y = selected_slice[:, 1]

        # Filter out points that are more than 150 units away from the previous point
        """filtered_x = [x[0]]
        filtered_y = [y[0]]

        for i in range(1, len(x)):
            distance = np.sqrt((x[i] - filtered_x[-1]) ** 2 + (y[i] - filtered_y[-1]) ** 2)
            if distance <= 150:
                filtered_x.append(x[i])
                filtered_y.append(y[i])

        filtered_x = np.array(filtered_x)
        filtered_y = np.array(filtered_y)"""

        # Sort by x while preserving the associated y values
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        # Find the index of the maximum value in filtered_y
        max_index = np.argmax(y_sorted)

        # Split filtered_x and filtered_y into two parts at the index of the maximum y
        x1, x2 = x_sorted[:max_index + 1], x_sorted[max_index:]
        y1, y2 = y_sorted[:max_index + 1], y_sorted[max_index:]

        # Linear regression 1
        model1 = LinearRegression()
        model1.fit(x1.reshape(-1, 1), y1)
        y_pred1 = model1.predict(x1.reshape(-1, 1))
        m1 = model1.coef_[0]

        # Linear regression 2
        model2 = LinearRegression()
        model2.fit(x2.reshape(-1, 1), y2)
        y_pred2 = model2.predict(x2.reshape(-1, 1))
        m2 = model2.coef_[0]

        # Calculate the angle between the two lines
        angle_rad = np.arctan(abs((m1 - m2) / (1 + m1 * m2)))
        angle_deg = np.degrees(angle_rad)
        self.ang_val.setText(f"{angle_deg:.2f}".replace('.', ','))

        # Show plot
        self.ax5.clear()
        self.ax5.set_xlabel('x [mm]')
        self.ax5.set_ylabel('y [mm]')
        self.ax5.scatter(x1, y1, color='blue', label='Actual left side')
        self.ax5.plot(x1, y_pred1, color='lightblue', label='Predicted left side', linewidth=2)
        self.ax5.scatter(x2, y2, color='red', label='Actual right side')
        self.ax5.plot(x2, y_pred2, color='pink', label='Predicted right side', linewidth=2)
        self.ax5.legend()
        self.ax5.text(0.05, 0.95, f'Angle: {angle_deg:.2f} degrees', transform=self.ax5.transAxes, fontsize=12,
                      verticalalignment='top')
        self.ax5.set_title(f'Linear regression of two sides of an L-shaped profile\n and the angle between them')
        self.canvas5.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
