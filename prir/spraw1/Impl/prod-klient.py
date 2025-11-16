"""
Producer-Consumer GUI (PySide6)

Features:
- PySide6 GUI with Start / Stop / Reset
- Mode selection: Sequential or Multithreaded
- Animated buffer visualization (rectangles)
- Controls: number of producers/consumers, production/consumption delay
- Live statistics and matplotlib chart embedded
- Simple benchmark (ops/sec) and timeline

Requirements:
- Python 3.9+
- pip install PySide6 matplotlib

Run:
    python producer_consumer_pyqt.py

"""
import sys
import time
import threading
import random
from queue import Queue, Empty
from collections import deque

from PySide6.QtCore import Qt, QTimer, QRectF, Slot
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSpinBox, QSlider, QFrame, QGridLayout, QSizePolicy
)
from PySide6.QtGui import QPainter, QColor, QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class BufferWidget(QFrame):
    """Simple visual of a bounded buffer using rectangles."""
    def __init__(self, capacity=10, parent=None):
        super().__init__(parent)
        self.capacity = capacity
        self.items = 0
        self.setMinimumHeight(80)

    def set_state(self, items, capacity=None):
        self.items = items
        if capacity is not None:
            self.capacity = capacity
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect().adjusted(10, 10, -10, -10)
        painter.setPen(Qt.black)
        painter.drawRect(rect)

        if self.capacity <= 0:
            return

        cell_w = rect.width() / self.capacity
        for i in range(self.capacity):
            cell_rect = QRectF(rect.left() + i * cell_w, rect.top(), cell_w - 2, rect.height())
            if i < self.items:
                painter.fillRect(cell_rect, QColor(70, 130, 180))
            else:
                painter.fillRect(cell_rect, QColor(230, 230, 230))
            painter.drawRect(cell_rect)


class StatsPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(5, 2.5), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.times = deque(maxlen=200)
        self.produced = deque(maxlen=200)
        self.consumed = deque(maxlen=200)
        self.start_time = None
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def add_point(self, t, p, c):
        # store relative time (seconds since start)
        if self.start_time is None:
            self.start_time = t
        self.times.append(t - self.start_time)
        self.produced.append(p)
        self.consumed.append(c)

    def redraw(self):
        self.ax.clear()
        if self.times:
            self.ax.plot(self.times, self.produced, label='Produced/s')
            self.ax.plot(self.times, self.consumed, label='Consumed/s')
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Operations per second')
            self.ax.legend(loc='upper left')
        else:
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Operations per second')
        self.canvas.draw()


class ProducerConsumerController:
    def __init__(self):
        self.mode = 'Multithreaded'  # or 'Sequential'
        self.buffer_capacity = 20
        self.queue = Queue(maxsize=self.buffer_capacity)
        self.producer_threads = []
        self.consumer_threads = []
        self.running = False
        self.lock = threading.Lock()
        self.rate_updated_callback = None

        # runtime stats
        self.total_produced = 0
        self.total_consumed = 0
        self.last_produced = 0
        self.last_consumed = 0

        # rates history (per-second)
        self.ops_produced = 0
        self.ops_consumed = 0

        # dynamic delays (threads read these each loop)
        self.prod_delay = 0.01
        self.cons_delay = 0.01

    def reset_stats(self):
        with self.lock:
            self.total_produced = 0
            self.total_consumed = 0
            self.last_produced = 0
            self.last_consumed = 0
            self.ops_produced = 0
            self.ops_consumed = 0

    def set_capacity(self, cap):
        self.buffer_capacity = cap
        # create a new queue to reflect capacity change (drains old content)
        self.queue = Queue(maxsize=self.buffer_capacity)

    def start(self, producers=1, consumers=1, prod_delay=0.01, cons_delay=0.01):
        # ensure any prior run is stopped
        self.stop()
        self.running = True
        self.reset_stats()
        self.queue = Queue(maxsize=self.buffer_capacity)

        # update dynamic delays
        with self.lock:
            self.prod_delay = prod_delay
            self.cons_delay = cons_delay

        # clear old thread lists
        self.producer_threads = []
        self.consumer_threads = []

        if self.mode == 'Sequential':
            # sequential loop runs in a separate thread to avoid blocking GUI
            self._seq_thread = threading.Thread(target=self._run_sequential,
                                                args=(producers, consumers, prod_delay, cons_delay),
                                                daemon=True)
            self._seq_thread.start()
        else:
            # spawn producer threads
            self._stop_event = threading.Event()
            for _ in range(producers):
                t = threading.Thread(target=self._producer_worker,
                                     args=(self._stop_event,), daemon=True)
                self.producer_threads.append(t)
                t.start()
            for _ in range(consumers):
                t = threading.Thread(target=self._consumer_worker,
                                     args=(self._stop_event,), daemon=True)
                self.consumer_threads.append(t)
                t.start()

            # rate counter thread
            self._rate_thread = threading.Thread(target=self._rate_counter, args=(self._stop_event,), daemon=True)
            self._rate_thread.start()

    def stop(self):
        self.running = False
        try:
            if hasattr(self, '_stop_event'):
                self._stop_event.set()
        except Exception:
            pass
        # join threads with a small timeout to avoid blocking UI indefinitely
        all_threads = []
        all_threads.extend(getattr(self, 'producer_threads', []))
        all_threads.extend(getattr(self, 'consumer_threads', []))
        for t in all_threads:
            try:
                if t.is_alive():
                    t.join(timeout=0.5)
            except RuntimeError:
                pass

        # clear lists
        self.producer_threads = []
        self.consumer_threads = []

    def _producer_worker(self, stop_event):
        while not stop_event.is_set():
            item = random.randint(1, 1000)
            try:
                # try to put; if full, this will wait up to 0.1s
                self.queue.put(item, timeout=0.1)
                with self.lock:
                    self.total_produced += 1
                    self.ops_produced += 1
            except Exception:
                # queue full or other; just continue
                pass
            # read dynamic delay
            with self.lock:
                d = self.prod_delay
            time.sleep(d)

    def _consumer_worker(self, stop_event):
        while not stop_event.is_set():
            try:
                self.queue.get(timeout=0.1)
                with self.lock:
                    self.total_consumed += 1
                    self.ops_consumed += 1
            except Empty:
                pass
            # read dynamic delay
            with self.lock:
                d = self.cons_delay
            time.sleep(d)

    def _run_sequential(self, producers, consumers, prod_delay, cons_delay):
        # run a tight loop producing then consuming in sequence
        while self.running:
            # produce producers items
            for _ in range(producers):
                if not self.queue.full():
                    self.queue.put(random.randint(1, 1000))
                    with self.lock:
                        self.total_produced += 1
                        self.ops_produced += 1
                time.sleep(prod_delay)
            # consume consumers items
            for _ in range(consumers):
                if not self.queue.empty():
                    try:
                        self.queue.get_nowait()
                        with self.lock:
                            self.total_consumed += 1
                            self.ops_consumed += 1
                    except Empty:
                        pass
                time.sleep(cons_delay)

    def _rate_counter(self, stop_event):
        # per-second rate updater
        while not stop_event.is_set():
            time.sleep(1.0)
            with self.lock:
                self.last_produced = self.ops_produced
                self.last_consumed = self.ops_consumed
                self.ops_produced = 0
                self.ops_consumed = 0

            if self.rate_updated_callback:
                self.rate_updated_callback()

    def snapshot(self):
        with self.lock:
            return {
                'produced': self.total_produced,
                'consumed': self.total_consumed,
                'last_produced': self.last_produced,
                'last_consumed': self.last_consumed,
                'buffer_items': self.queue.qsize(),
                'capacity': self.buffer_capacity
            }


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Producer-Consumer – PySide6 Demo')
        self.controller = ProducerConsumerController()

        self._build_ui()
        self._make_timer()
        self.controller.rate_updated_callback = self._rate_updated

    def _rate_updated(self):
        snap = self.controller.snapshot()
        now = time.time()
        self.plot.add_point(now, snap['last_produced'], snap['last_consumed'])
        self.plot.redraw()

    def _build_ui(self):
        main = QVBoxLayout(self)

        # Controls
        ctrl = QGridLayout()

        ctrl.addWidget(QLabel('Mode:'), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Multithreaded', 'Sequential'])
        self.mode_combo.currentTextChanged.connect(self._on_mode_change)
        ctrl.addWidget(self.mode_combo, 0, 1)

        ctrl.addWidget(QLabel('Buffer capacity:'), 1, 0)
        self.capacity_spin = QSpinBox()
        self.capacity_spin.setRange(1, 100)
        self.capacity_spin.setValue(self.controller.buffer_capacity)
        ctrl.addWidget(self.capacity_spin, 1, 1)

        ctrl.addWidget(QLabel('Producers:'), 2, 0)
        self.producers_spin = QSpinBox()
        # allow more producers than before to stress test system
        self.producers_spin.setRange(1, 200)
        self.producers_spin.setValue(2)
        ctrl.addWidget(self.producers_spin, 2, 1)

        ctrl.addWidget(QLabel('Consumers:'), 3, 0)
        self.consumers_spin = QSpinBox()
        # allow more consumers
        self.consumers_spin.setRange(1, 200)
        self.consumers_spin.setValue(2)
        ctrl.addWidget(self.consumers_spin, 3, 1)

        ctrl.addWidget(QLabel('Producer delay (ms):'), 4, 0)
        self.prod_delay_slider = QSlider(Qt.Horizontal)
        self.prod_delay_slider.setRange(0, 2000)  # up to 2s delay
        self.prod_delay_slider.setValue(10)
        ctrl.addWidget(self.prod_delay_slider, 4, 1)

        ctrl.addWidget(QLabel('Consumer delay (ms):'), 5, 0)
        self.cons_delay_slider = QSlider(Qt.Horizontal)
        self.cons_delay_slider.setRange(0, 2000)
        self.cons_delay_slider.setValue(10)
        ctrl.addWidget(self.cons_delay_slider, 5, 1)

        main.addLayout(ctrl)

        # Buttons
        btns = QHBoxLayout()
        self.start_btn = QPushButton('Start')
        self.start_btn.clicked.connect(self.start)
        btns.addWidget(self.start_btn)

        self.stop_btn = QPushButton('Stop')
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        btns.addWidget(self.stop_btn)

        self.reset_btn = QPushButton('Reset')
        self.reset_btn.clicked.connect(self.reset)
        btns.addWidget(self.reset_btn)

        main.addLayout(btns)

        # Buffer visualization and stats
        vis_layout = QHBoxLayout()
        self.buffer_widget = BufferWidget(capacity=self.controller.buffer_capacity)
        vis_layout.addWidget(self.buffer_widget, 2)

        stats_col = QVBoxLayout()
        self.lbl_buffer = QLabel('Buffer: 0 / 0')
        self.lbl_totals = QLabel('Produced: 0    Consumed: 0')
        self.lbl_rate = QLabel('Rate (last sec): P=0 C=0')
        stats_col.addWidget(self.lbl_buffer)
        stats_col.addWidget(self.lbl_totals)
        stats_col.addWidget(self.lbl_rate)
        vis_layout.addLayout(stats_col, 1)

        main.addLayout(vis_layout)

        # Plot
        self.plot = StatsPlot()
        main.addWidget(self.plot)

        self.setLayout(main)

    def _on_mode_change(self, text):
        self.controller.mode = text

    def _make_timer(self):
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(250)  # update UI 4x/s
        self.ui_timer.timeout.connect(self._update_ui)
        self.ui_timer.start()

        # per-second chart timer
        # self.chart_timer = QTimer(self)
        # self.chart_timer.setInterval(1000)
        # self.chart_timer.timeout.connect(self._update_chart)
        # self.chart_timer.start()

    @Slot()
    def start(self):
        cap = self.capacity_spin.value()
        self.controller.set_capacity(cap)
        producers = self.producers_spin.value()
        consumers = self.consumers_spin.value()
        prod_delay = max(0.0, self.prod_delay_slider.value() / 1000)
        cons_delay = max(0.0, self.cons_delay_slider.value() / 1000)
        # update controller dynamic delays so running threads pick them up
        with self.controller.lock:
            self.controller.prod_delay = prod_delay
            self.controller.cons_delay = cons_delay
        self.controller.start(producers=producers, consumers=consumers,
                              prod_delay=prod_delay, cons_delay=cons_delay)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    @Slot()
    def stop(self):
        self.controller.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    @Slot()
    def reset(self):
        self.controller.stop()
        self.controller.reset_stats()
        self.controller.queue = Queue(maxsize=self.controller.buffer_capacity)
        self.plot.start_time = None
        self._update_ui()

    def _update_ui(self):
        snap = self.controller.snapshot()
        self.buffer_widget.set_state(snap['buffer_items'], snap['capacity'])
        self.lbl_buffer.setText(f"Buffer: {snap['buffer_items']} / {snap['capacity']}")
        self.lbl_totals.setText(f"Produced: {snap['produced']}    Consumed: {snap['consumed']}")
        self.lbl_rate.setText(f"Rate (last sec): P={snap['last_produced']} C={snap['last_consumed']}")

    def _update_chart(self):
        now = time.time()
        snap = self.controller.snapshot()
        self.plot.add_point(now, snap['last_produced'], snap['last_consumed'])
        self.plot.redraw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(900, 700)
    w.show()
    sys.exit(app.exec())
