#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
controller_gui.py

Graphical control panel for Duckiedrone DD21 validation experiments.

Stage 1.2 features (UI sophistication upgrade, Steps 1+2):
- Start / stop Gazebo + DD21.
- Start / stop one controller/scenario launch.
- Select controller, scenario, run ID, mismatch, and odometry bridge.
- Monitor ROS master, Gazebo, odometry, controller, and motor-command status.
- Dark engineering theme (QSS), header bar with global system state,
  two-column card layout, LED-style status dashboard.
- Live roslaunch command preview, auto-increment Run ID option,
  GUI-owned process runtime readouts.
- Console with timestamps, color-coded tags, tag filters, text search,
  autoscroll toggle and save-to-file.
- Shortcuts: Ctrl+L clear console, Ctrl+S save log, Ctrl+F focus search.

The GUI does not modify PID/MPC YAML files in this stage.

Author: Abdallah GHOUL
"""

import html
import os
import signal
import subprocess
import sys
import time
from typing import Optional, Set, Tuple

import rosgraph

from PyQt5.QtCore import QProcess, QTimer, Qt
from PyQt5.QtGui import QFont, QIcon, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSpinBox,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


WORKSPACE = os.path.expanduser("~/duckiedrone_sim")
ROS_SETUP = "/opt/ros/noetic/setup.bash"
WS_SETUP = os.path.join(WORKSPACE, "devel", "setup.bash")

# Application icon: expected next to this script. The GUI works normally
# even if the file is missing (all uses are guarded).
ICON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "duckiedrone_icon.png"
)

GAZEBO_LAUNCH = "roslaunch duckiedrone_description spawn_dd21.launch"

CONTROLLER_NODES = {
    "pid": "/pid_controller",
    "mpc": "/mpc_controller",
    "physics_mpc": "/physics_mpc_controller",
    "pem_mpc": "/pem_mpc_controller",
    "vstmpc": "/vstmpc_controller",
}

# ----------------------------------------------------------------------
# Theme tokens
# ----------------------------------------------------------------------

COLOR_OK = "#4caf7d"
COLOR_ERR = "#cf6a6a"
COLOR_WARN = "#d9a441"
COLOR_DIM = "#5b636d"
COLOR_ACCENT = "#3aa79b"

TAG_COLORS = {
    "GAZEBO": COLOR_WARN,
    "CONTROLLER": COLOR_ACCENT,
    "GUI": "#98a1ab",
}

CONSOLE_TAGS = ("ALL", "GAZEBO", "CONTROLLER", "GUI")

STYLE_SHEET = """
QMainWindow, QWidget {
    background-color: #26292f;
    color: #d8dde4;
    font-family: "Segoe UI", "Ubuntu", "Cantarell", "Noto Sans", sans-serif;
    font-size: 10pt;
}

QLabel {
    background-color: transparent;
}
QCheckBox {
    background-color: transparent;
}

/* ---------- Cards (QGroupBox) ---------- */
QGroupBox {
    background-color: #2e323a;
    border: 1px solid #3c424c;
    border-radius: 8px;
    margin-top: 16px;
    padding-top: 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #9fb0bc;
}

/* ---------- Header bar ---------- */
QFrame#headerBar {
    background-color: #20232a;
    border: 1px solid #343a44;
    border-radius: 8px;
}
QLabel#appTitle {
    font-size: 15pt;
    font-weight: 700;
    color: #eef2f6;
}
QLabel#appSubtitle {
    color: #8b96a1;
    font-size: 9pt;
}
QLabel#systemChip {
    border-radius: 10px;
    padding: 5px 14px;
    font-weight: 700;
}

/* ---------- Buttons ---------- */
QPushButton {
    background-color: #3a404a;
    border: 1px solid #4a5260;
    border-radius: 6px;
    padding: 7px 16px;
    color: #e2e7ec;
}
QPushButton:hover {
    background-color: #454d59;
    border-color: #5a6472;
}
QPushButton:pressed {
    background-color: #31373f;
}
QPushButton:disabled {
    background-color: #2b2f36;
    color: #6b7480;
    border-color: #383e47;
}
QPushButton#primaryButton {
    background-color: #2e7d6e;
    border-color: #379786;
    color: #f0f7f5;
    font-weight: 600;
}
QPushButton#primaryButton:hover {
    background-color: #35907f;
    border-color: #42ab98;
}
QPushButton#primaryButton:pressed {
    background-color: #27695d;
}
QPushButton#primaryButton:disabled {
    background-color: #2b3a37;
    color: #637a75;
    border-color: #354642;
}
QPushButton#dangerButton {
    background-color: #7d3a3a;
    border-color: #944646;
    color: #f7f0f0;
    font-weight: 600;
}
QPushButton#dangerButton:hover {
    background-color: #904444;
    border-color: #a85252;
}
QPushButton#dangerButton:pressed {
    background-color: #693232;
}
QPushButton#dangerButton:disabled {
    background-color: #3a2f2f;
    color: #7a6b6b;
    border-color: #463a3a;
}
QPushButton#chipButton {
    background-color: #2b2f36;
    border: 1px solid #3c424c;
    border-radius: 10px;
    padding: 3px 12px;
    color: #98a1ab;
    font-weight: 600;
}
QPushButton#chipButton:hover:!checked {
    background-color: #3a404a;
}
QPushButton#chipButton:checked {
    background-color: #2e7d6e;
    border-color: #379786;
    color: #f0f7f5;
}

/* ---------- Inputs ---------- */
QComboBox, QSpinBox, QLineEdit {
    background-color: #24282e;
    border: 1px solid #424957;
    border-radius: 5px;
    padding: 5px 8px;
    color: #d8dde4;
    min-height: 20px;
}
QComboBox:hover, QSpinBox:hover, QLineEdit:hover {
    border-color: #3aa79b;
}
QComboBox:focus, QSpinBox:focus, QLineEdit:focus {
    border-color: #3aa79b;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 22px;
    border-left: 1px solid #424957;
    border-top-right-radius: 5px;
    border-bottom-right-radius: 5px;
}
QComboBox QAbstractItemView {
    background-color: #24282e;
    color: #d8dde4;
    border: 1px solid #424957;
    selection-background-color: #2e7d6e;
    selection-color: #f0f7f5;
    outline: none;
}
QSpinBox::up-button, QSpinBox::down-button {
    background-color: #3a404a;
    border: none;
    width: 18px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background-color: #454d59;
}
QLineEdit#commandPreview {
    background-color: #1d2025;
    color: #9fb7ae;
}

QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #4a5260;
    background-color: #24282e;
}
QCheckBox::indicator:hover {
    border-color: #3aa79b;
}
QCheckBox::indicator:checked {
    background-color: #2e7d6e;
    border-color: #379786;
}

/* ---------- Console ---------- */
QTextEdit {
    background-color: #1d2025;
    color: #c9d1d9;
    border: 1px solid #3c424c;
    border-radius: 6px;
    selection-background-color: #2e7d6e;
}

/* ---------- Status dashboard ---------- */
QLabel[class="statusName"] {
    color: #98a1ab;
}
QLabel[class="statusValue"] {
    color: #d8dde4;
    font-weight: 600;
}

/* ---------- Misc ---------- */
QStatusBar {
    background-color: #20232a;
    color: #98a1ab;
}
QToolTip {
    background-color: #2e323a;
    color: #d8dde4;
    border: 1px solid #4a5260;
    padding: 4px;
}
QMessageBox QLabel {
    color: #d8dde4;
}
"""


class DuckiedroneControlPanel(QMainWindow):
    """Main Duckiedrone experiment control window."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Duckiedrone DD21 - Control Panel")
        self.resize(1080, 860)

        if os.path.isfile(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))

        self.gazebo_process = self._create_process("GAZEBO")
        self.controller_process = self._create_process("CONTROLLER")
        self._gazebo_cleanup_process = None

        # GUI-owned process runtimes (monotonic timestamps, None = not running)
        self._gazebo_started_at: Optional[float] = None
        self._controller_started_at: Optional[float] = None

        self.gazebo_process.started.connect(
            lambda: setattr(self, "_gazebo_started_at", time.monotonic())
        )
        self.gazebo_process.finished.connect(
            lambda *args: setattr(self, "_gazebo_started_at", None)
        )
        self.controller_process.started.connect(
            lambda: setattr(self, "_controller_started_at", time.monotonic())
        )
        self.controller_process.finished.connect(
            lambda *args: setattr(self, "_controller_started_at", None)
        )
        self.controller_process.started.connect(self._auto_increment_run_id)

        # Full in-memory console log: list of (timestamp, tag, message)
        self._log_entries = []
        self._active_filter = "ALL"

        self._build_ui()
        self._setup_shortcuts()

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_ros_status)
        self.status_timer.start(1000)

        self.update_ros_status()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(14, 14, 14, 10)
        main_layout.setSpacing(12)

        main_layout.addWidget(self._build_header_bar())

        body = QHBoxLayout()
        body.setSpacing(12)

        left_column = QVBoxLayout()
        left_column.setSpacing(12)
        left_column.addWidget(self._build_simulation_group())
        left_column.addWidget(self._build_controller_group())
        left_column.addStretch()

        right_column = QVBoxLayout()
        right_column.setSpacing(12)
        right_column.addWidget(self._build_status_group())
        right_column.addStretch()

        body.addLayout(left_column, stretch=3)
        body.addLayout(right_column, stretch=2)

        main_layout.addLayout(body)
        main_layout.addWidget(self._build_console_group(), stretch=1)

        self.statusBar().showMessage(
            "Stage 1: process control + ROS status. Controller gains remain unchanged."
        )

    def _setup_shortcuts(self):
        clear_sc = QShortcut(QKeySequence("Ctrl+L"), self)
        clear_sc.activated.connect(self.console_clear)

        save_sc = QShortcut(QKeySequence("Ctrl+S"), self)
        save_sc.activated.connect(self.console_save)

        search_sc = QShortcut(QKeySequence("Ctrl+F"), self)
        search_sc.activated.connect(
            lambda: self.search_box.setFocus(Qt.ShortcutFocusReason)
        )

    def _build_header_bar(self) -> QFrame:
        header = QFrame(self)
        header.setObjectName("headerBar")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 10, 16, 10)

        if os.path.isfile(ICON_PATH):
            icon_label = QLabel()
            icon_label.setPixmap(
                QPixmap(ICON_PATH).scaled(
                    44, 44, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            icon_label.setFixedSize(44, 44)
            layout.addWidget(icon_label)
            layout.addSpacing(6)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)

        title = QLabel("Duckiedrone DD21 — Controllers Panel")
        title.setObjectName("appTitle")

        subtitle = QLabel(
            "Simulation · PID / MPC / Physics-MPC / PEM-MPC / VS-TMPC · "
            "Scenarios S1–S5"
        )
        subtitle.setObjectName("appSubtitle")

        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.system_chip = QLabel("IDLE")
        self.system_chip.setObjectName("systemChip")
        self.system_chip.setAlignment(Qt.AlignCenter)
        self.system_chip.setToolTip(
            "Global system state:\n"
            "IDLE — no simulation running\n"
            "SIMULATION READY — Gazebo online, no controller\n"
            "EXPERIMENT RUNNING — controller node active"
        )

        layout.addLayout(title_box)
        layout.addStretch()
        layout.addWidget(self.system_chip)

        return header

    def _build_simulation_group(self) -> QGroupBox:
        group = QGroupBox("SIMULATION")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(10)

        self.gazebo_state_label = QLabel("Gazebo: UNKNOWN")
        self.gazebo_state_label.setMinimumWidth(230)
        self.gazebo_state_label.setStyleSheet(
            "QLabel { font-weight: 600; color: #98a1ab; }"
        )

        style = self.style()
        self.start_gazebo_button = QPushButton(
            style.standardIcon(QStyle.SP_MediaPlay), "Start Gazebo"
        )
        self.stop_gazebo_button = QPushButton(
            style.standardIcon(QStyle.SP_MediaStop), "Stop Gazebo"
        )
        self.start_gazebo_button.setObjectName("primaryButton")
        self.stop_gazebo_button.setObjectName("dangerButton")
        self.start_gazebo_button.setToolTip(
            "roslaunch duckiedrone_description spawn_dd21.launch"
        )
        self.stop_gazebo_button.setToolTip(
            "Stop the GUI-owned Gazebo launch (controller must be stopped first)."
        )

        self.start_gazebo_button.clicked.connect(self.start_gazebo)
        self.stop_gazebo_button.clicked.connect(self.stop_gazebo)

        layout.addWidget(self.gazebo_state_label)
        layout.addStretch()
        layout.addWidget(self.start_gazebo_button)
        layout.addWidget(self.stop_gazebo_button)

        return group

    def _build_controller_group(self) -> QGroupBox:
        group = QGroupBox("CONTROLLER / SCENARIO")
        outer = QVBoxLayout(group)
        outer.setContentsMargins(12, 8, 12, 12)
        outer.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)

        self.controller_combo = QComboBox()
        self.controller_combo.addItem("PID", "pid")
        self.controller_combo.addItem("MPC", "mpc")
        self.controller_combo.addItem("Physics MPC", "physics_mpc")
        self.controller_combo.addItem("PEM-MPC", "pem_mpc")
        self.controller_combo.addItem("VS-TMPC", "vstmpc")

        self.scenario_combo = QComboBox()
        for scenario in (
            "S1",
            "S2-Roll",
            "S2-Pitch",
            "S2-Yaw",
            "S3",
            "S4",
            "S5",
        ):
            self.scenario_combo.addItem(scenario, scenario)
        self.scenario_combo.currentTextChanged.connect(self._scenario_changed)

        self.run_id_spin = QSpinBox()
        self.run_id_spin.setRange(1, 9999)
        self.run_id_spin.setValue(1)

        self.mismatch_checkbox = QCheckBox("Use mismatched controller parameters")
        self.use_bridge_checkbox = QCheckBox("Use /gazebo/model_states -> /odom bridge")
        self.use_bridge_checkbox.setChecked(True)

        self.auto_increment_checkbox = QCheckBox(
            "Auto-increment Run ID after successful start"
        )
        self.auto_increment_checkbox.setChecked(True)
        self.auto_increment_checkbox.setToolTip(
            "When checked, Run ID increases by 1 every time a controller "
            "launch actually starts, so consecutive runs never overwrite "
            "each other's logs."
        )

        form.addRow("Controller:", self.controller_combo)
        form.addRow("Scenario:", self.scenario_combo)
        form.addRow("Run ID:", self.run_id_spin)
        form.addRow("Mismatch:", self.mismatch_checkbox)
        form.addRow("Odometry bridge:", self.use_bridge_checkbox)
        form.addRow("Run counter:", self.auto_increment_checkbox)

        outer.addLayout(form)

        preview_caption = QLabel("Launch command preview:")
        preview_caption.setProperty("class", "statusName")
        self.command_preview = QLineEdit()
        self.command_preview.setObjectName("commandPreview")
        self.command_preview.setReadOnly(True)
        preview_font = QFont("Monospace")
        preview_font.setStyleHint(QFont.TypeWriter)
        preview_font.setPointSize(8)
        self.command_preview.setFont(preview_font)
        self.command_preview.setToolTip(
            "The exact roslaunch command that will be executed with the "
            "current selection."
        )
        outer.addWidget(preview_caption)
        outer.addWidget(self.command_preview)

        self.controller_combo.currentIndexChanged.connect(
            self._update_command_preview
        )
        self.scenario_combo.currentTextChanged.connect(
            self._update_command_preview
        )
        self.run_id_spin.valueChanged.connect(self._update_command_preview)
        self.mismatch_checkbox.toggled.connect(self._update_command_preview)
        self.use_bridge_checkbox.toggled.connect(self._update_command_preview)
        self._update_command_preview()

        button_row = QHBoxLayout()
        style = self.style()
        self.start_controller_button = QPushButton(
            style.standardIcon(QStyle.SP_MediaPlay), "Start Controller"
        )
        self.stop_controller_button = QPushButton(
            style.standardIcon(QStyle.SP_MediaStop), "Stop Controller"
        )
        self.start_controller_button.setObjectName("primaryButton")
        self.stop_controller_button.setObjectName("dangerButton")

        self.start_controller_button.clicked.connect(self.start_controller)
        self.stop_controller_button.clicked.connect(self.stop_controller)

        button_row.addStretch()
        button_row.addWidget(self.start_controller_button)
        button_row.addWidget(self.stop_controller_button)

        outer.addLayout(button_row)
        return group

    def _build_status_group(self) -> QGroupBox:
        group = QGroupBox("ROS STATUS")
        layout = QGridLayout(group)
        layout.setContentsMargins(14, 10, 14, 14)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(12)
        layout.setColumnStretch(2, 1)

        rows = [
            ("ROS Master", "ROS Master reachable (roscore)."),
            ("Gazebo", "/gazebo node registered with the master."),
            ("/odom", "Odometry topic has an active publisher."),
            ("Controller", "Controller node registered with the master."),
            ("/cmd_motors", "Motor-command topic has an active publisher."),
            ("Gazebo run", "Wall-clock time the GUI-owned Gazebo launch "
                           "has been running."),
            ("Ctrl run", "Wall-clock time the GUI-owned controller/scenario "
                         "launch has been running."),
        ]

        self.master_led = self._make_led()
        self.gazebo_led = self._make_led()
        self.odom_led = self._make_led()
        self.controller_led = self._make_led()
        self.motors_led = self._make_led()
        self.gazebo_runtime_led = self._make_led()
        self.controller_runtime_led = self._make_led()

        self.master_status = QLabel("UNKNOWN")
        self.gazebo_status = QLabel("UNKNOWN")
        self.odom_status = QLabel("UNKNOWN")
        self.controller_status = QLabel("UNKNOWN")
        self.motors_status = QLabel("UNKNOWN")
        self.gazebo_runtime = QLabel("—")
        self.controller_runtime = QLabel("—")

        leds = [
            self.master_led,
            self.gazebo_led,
            self.odom_led,
            self.controller_led,
            self.motors_led,
            self.gazebo_runtime_led,
            self.controller_runtime_led,
        ]
        values = [
            self.master_status,
            self.gazebo_status,
            self.odom_status,
            self.controller_status,
            self.motors_status,
            self.gazebo_runtime,
            self.controller_runtime,
        ]

        for row, ((name, tip), led, value) in enumerate(zip(rows, leds, values)):
            name_label = QLabel(name)
            name_label.setProperty("class", "statusName")
            value.setProperty("class", "statusValue")
            name_label.setToolTip(tip)
            value.setToolTip(tip)
            led.setToolTip(tip)
            layout.addWidget(led, row, 0, Qt.AlignVCenter)
            layout.addWidget(name_label, row, 1)
            layout.addWidget(value, row, 2, Qt.AlignRight)

        return group

    @staticmethod
    def _make_led() -> QLabel:
        led = QLabel()
        led.setFixedSize(12, 12)
        led.setStyleSheet(
            "QLabel { background-color: %s; border-radius: 6px; }" % COLOR_DIM
        )
        return led

    def _set_led(self, led: QLabel, color: str):
        led.setStyleSheet(
            "QLabel { background-color: %s; border-radius: 6px; }" % color
        )

    def _build_console_group(self) -> QGroupBox:
        group = QGroupBox("CONSOLE")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self._filter_group = QButtonGroup(self)
        self._filter_group.setExclusive(True)
        self._filter_buttons = {}
        for tag in CONSOLE_TAGS:
            button = QPushButton(tag)
            button.setObjectName("chipButton")
            button.setCheckable(True)
            button.setChecked(tag == "ALL")
            button.setFixedHeight(26)
            button.setToolTip(f"Show {tag} messages" if tag != "ALL"
                              else "Show all messages")
            self._filter_group.addButton(button)
            self._filter_buttons[tag] = button
            controls.addWidget(button)
        self._filter_group.buttonClicked.connect(self._console_filter_changed)

        controls.addSpacing(12)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search console…  (Ctrl+F)")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.setFixedHeight(28)
        self.search_box.setMaximumWidth(260)
        self.search_box.textChanged.connect(self._refresh_console)
        controls.addWidget(self.search_box)

        controls.addStretch()

        self.autoscroll_checkbox = QCheckBox("Autoscroll")
        self.autoscroll_checkbox.setChecked(True)
        controls.addWidget(self.autoscroll_checkbox)

        save_button = QPushButton("Save Log")
        save_button.setFixedHeight(28)
        save_button.setToolTip("Save the full console log to a file (Ctrl+S).")
        save_button.clicked.connect(self.console_save)
        controls.addWidget(save_button)

        clear_button = QPushButton("Clear Console")
        clear_button.setFixedHeight(28)
        clear_button.setToolTip("Clear the console view and log (Ctrl+L).")
        clear_button.clicked.connect(self.console_clear)
        controls.addWidget(clear_button)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.document().setMaximumBlockCount(5000)

        font = QFont("Monospace")
        font.setStyleHint(QFont.TypeWriter)
        font.setPointSize(9)
        self.console.setFont(font)

        layout.addLayout(controls)
        layout.addWidget(self.console)

        return group

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    def _create_process(self, tag: str) -> QProcess:
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)

        process.readyReadStandardOutput.connect(
            lambda p=process, t=tag: self._read_process_output(p, t)
        )
        process.started.connect(
            lambda t=tag: self.append_console(t, "Process started.")
        )
        process.finished.connect(
            lambda exit_code, exit_status, t=tag:
            self.append_console(
                t,
                f"Process finished: exit_code={exit_code}, "
                f"exit_status={int(exit_status)}"
            )
        )
        process.errorOccurred.connect(
            lambda error, t=tag:
            self.append_console(t, f"QProcess error: {int(error)}")
        )

        return process

    def _start_ros_command(self, process: QProcess, command: str, tag: str):
        if process.state() != QProcess.NotRunning:
            self.append_console(tag, "Process is already running.")
            return

        if not os.path.isfile(WS_SETUP):
            QMessageBox.critical(
                self,
                "Workspace error",
                f"Cannot find workspace setup file:\n{WS_SETUP}",
            )
            return

        # exec makes the QProcess PID become the roslaunch PID. Gazebo itself
        # creates separate process groups for gzserver/gzclient, so those are
        # handled explicitly by stop_gazebo().
        shell_command = (
            f"source '{ROS_SETUP}' && "
            f"source '{WS_SETUP}' && "
            f"cd '{WORKSPACE}' && "
            f"exec {command}"
        )

        self.append_console(tag, f"$ {command}")
        process.setProgram("/bin/bash")
        process.setArguments(["-lc", shell_command])
        process.start()

    def _stop_process(self, process: QProcess, tag: str):
        """Gracefully stop a GUI-owned roslaunch process."""
        if process.state() == QProcess.NotRunning:
            self.append_console(tag, "No GUI-owned process is running.")
            return

        pid = int(process.processId())
        self.append_console(tag, f"Sending SIGINT to roslaunch PID {pid} ...")

        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            self.append_console(tag, "roslaunch already exited.")
            return
        except OSError as exc:
            self.append_console(tag, f"SIGINT failed: {exc}")
            process.terminate()

        QTimer.singleShot(
            5000,
            lambda p=process, t=tag: self._force_kill_if_needed(p, t),
        )

    def _force_kill_if_needed(self, process: QProcess, tag: str):
        if process.state() == QProcess.NotRunning:
            return

        self.append_console(
            tag,
            "roslaunch did not stop after 5 s; sending SIGTERM.",
        )
        process.terminate()

        QTimer.singleShot(
            2000,
            lambda p=process, t=tag: self._kill_if_still_running(p, t),
        )

    def _kill_if_still_running(self, process: QProcess, tag: str):
        if process.state() != QProcess.NotRunning:
            self.append_console(
                tag,
                "Process still running after SIGTERM; sending SIGKILL.",
            )
            process.kill()

    def _read_process_output(self, process: QProcess, tag: str):
        data = bytes(process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        for line in data.splitlines():
            self.append_console(tag, line)

    # ------------------------------------------------------------------
    # User actions
    # ------------------------------------------------------------------

    def start_gazebo(self):
        ros_online, nodes, _ = self._get_ros_state()

        if ros_online and "/gazebo" in nodes:
            QMessageBox.information(
                self,
                "Gazebo",
                "Gazebo is already running.\n"
                "The GUI will not start a second Gazebo instance.",
            )
            return

        self._start_ros_command(
            self.gazebo_process,
            GAZEBO_LAUNCH,
            "GAZEBO",
        )

    def stop_gazebo(self):
        """
        Stop the GUI-owned Gazebo launch cleanly.

        Gazebo 11 on this platform starts gzserver and gzclient in process
        groups that are different from the roslaunch process group. In
        particular, gzclient was observed to ignore SIGINT but terminate
        correctly on SIGTERM. Therefore Gazebo nodes/processes are stopped
        explicitly before roslaunch itself is terminated.
        """
        if self.controller_process.state() != QProcess.NotRunning:
            QMessageBox.warning(
                self,
                "Stop controller first",
                "Stop the controller before stopping Gazebo.",
            )
            return

        if self.gazebo_process.state() == QProcess.NotRunning:
            QMessageBox.information(
                self,
                "Gazebo",
                "Gazebo was not started by this GUI.\n"
                "For safety, external Gazebo processes are not killed.",
            )
            return

        self.stop_gazebo_button.setEnabled(False)
        self.append_console(
            "GAZEBO",
            "Stopping mixer and Gazebo ROS nodes..."
        )

        cleanup_command = (
            f"source '{ROS_SETUP}' && "
            f"source '{WS_SETUP}' && "
            "rosnode kill /mixer_node /gazebo_gui /gazebo "
            ">/dev/null 2>&1 || true; "
            "sleep 0.8; "
            "pkill -TERM -x gzclient >/dev/null 2>&1 || true; "
            "pkill -TERM -x gzserver >/dev/null 2>&1 || true; "
            "sleep 1.0; "
            "pkill -KILL -x gzclient >/dev/null 2>&1 || true; "
            "pkill -KILL -x gzserver >/dev/null 2>&1 || true"
        )

        self._gazebo_cleanup_process = QProcess(self)
        self._gazebo_cleanup_process.setProcessChannelMode(
            QProcess.MergedChannels
        )
        self._gazebo_cleanup_process.finished.connect(
            self._finish_gazebo_stop
        )
        self._gazebo_cleanup_process.start(
            "/bin/bash",
            ["-lc", cleanup_command],
        )

    def _finish_gazebo_stop(self, exit_code, exit_status):
        self.append_console(
            "GAZEBO",
            "Gazebo child processes stopped; stopping roslaunch..."
        )
        self._stop_process(self.gazebo_process, "GAZEBO")

    def _build_controller_command(self) -> str:
        """Build the roslaunch command from the current GUI selection."""
        controller = self.controller_combo.currentData()
        scenario_label = self.scenario_combo.currentText()

        # --------------------------------------------------
        # S2 is split into three independent attitude tests.
        #
        # GUI label    -> roslaunch arguments
        #
        # S2-Roll     -> scenario:=S2 s2_axis:=roll
        # S2-Pitch    -> scenario:=S2 s2_axis:=pitch
        # S2-Yaw      -> scenario:=S2 s2_axis:=yaw
        # --------------------------------------------------
        if scenario_label.startswith("S2-"):
            scenario = "S2"
            s2_axis = scenario_label.split("-", 1)[1].lower()
        else:
            scenario = scenario_label
            s2_axis = "roll"   # ignored unless scenario == S2

        run_id = self.run_id_spin.value()
        mismatch = self._bool_to_ros(self.mismatch_checkbox.isChecked())
        use_bridge = self._bool_to_ros(self.use_bridge_checkbox.isChecked())

        return (
            "roslaunch duckiedrone_validation run_scenario.launch "
            f"controller:={controller} "
            f"scenario:={scenario} "
            f"s2_axis:={s2_axis} "
            f"run_id:={run_id} "
            f"mismatch:={mismatch} "
            f"use_bridge:={use_bridge}"
        )

    def _update_command_preview(self):
        self.command_preview.setText(self._build_controller_command())
        self.command_preview.setCursorPosition(0)

    def _auto_increment_run_id(self):
        """Bump Run ID after a controller launch actually started."""
        if self.auto_increment_checkbox.isChecked():
            self.run_id_spin.stepUp()

    def start_controller(self):
        ros_online, nodes, _ = self._get_ros_state()

        if not ros_online:
            QMessageBox.warning(
                self,
                "ROS offline",
                "ROS Master is not available. Start Gazebo first.",
            )
            return

        if "/gazebo" not in nodes:
            QMessageBox.warning(
                self,
                "Gazebo offline",
                "Gazebo is not running. Start Gazebo first.",
            )
            return

        existing_controller = self._find_active_controller(nodes)
        if existing_controller:
            QMessageBox.warning(
                self,
                "Controller already running",
                f"An active controller was detected:\n{existing_controller}\n\n"
                "Stop it before starting another controller.",
            )
            return

        self._start_ros_command(
            self.controller_process,
            self._build_controller_command(),
            "CONTROLLER",
        )

    def stop_controller(self):
        if self.controller_process.state() == QProcess.NotRunning:
            ros_online, nodes, _ = self._get_ros_state()
            existing = self._find_active_controller(nodes) if ros_online else None

            if existing:
                QMessageBox.information(
                    self,
                    "External controller",
                    f"{existing} is running but was not started by this GUI.\n"
                    "For safety, the GUI will not terminate external processes.",
                )
            else:
                self.append_console(
                    "CONTROLLER",
                    "No GUI-owned controller process is running.",
                )
            return

        self._stop_process(self.controller_process, "CONTROLLER")

    def _scenario_changed(self, scenario: str):
        # S5 is the parametric-mismatch validation scenario in the current
        # run_scenario.launch design. The user may still override the checkbox.
        if scenario == "S5":
            self.mismatch_checkbox.setChecked(True)
        else:
            self.mismatch_checkbox.setChecked(False)

    # ------------------------------------------------------------------
    # ROS monitoring
    # ------------------------------------------------------------------

    def _get_ros_state(self) -> Tuple[bool, Set[str], Set[str]]:
        """
        Return:
            ros_online: ROS Master reachable
            nodes: all nodes seen in publishers/subscribers/services
            published_topics: topics that currently have publishers
        """
        try:
            master = rosgraph.Master("/duckiedrone_control_gui")
            master.getPid()

            publishers, subscribers, services = master.getSystemState()

            nodes: Set[str] = set()
            published_topics: Set[str] = set()

            for topic, topic_nodes in publishers:
                published_topics.add(topic)
                nodes.update(topic_nodes)

            for _, topic_nodes in subscribers:
                nodes.update(topic_nodes)

            for _, service_nodes in services:
                nodes.update(service_nodes)

            return True, nodes, published_topics

        except Exception:
            return False, set(), set()

    def update_ros_status(self):
        ros_online, nodes, published_topics = self._get_ros_state()

        self._set_status(
            self.master_status,
            self.master_led,
            ros_online,
            "ONLINE" if ros_online else "OFFLINE",
        )

        gazebo_online = "/gazebo" in nodes
        self._set_status(
            self.gazebo_status,
            self.gazebo_led,
            gazebo_online,
            "ONLINE" if gazebo_online else "OFFLINE",
        )

        odom_active = "/odom" in published_topics
        self._set_status(
            self.odom_status,
            self.odom_led,
            odom_active,
            "ACTIVE" if odom_active else "INACTIVE",
        )

        active_controller = self._find_active_controller(nodes)
        self._set_status(
            self.controller_status,
            self.controller_led,
            active_controller is not None,
            active_controller if active_controller else "STOPPED",
        )

        motors_active = "/cmd_motors" in published_topics
        self._set_status(
            self.motors_status,
            self.motors_led,
            motors_active,
            "ACTIVE" if motors_active else "INACTIVE",
        )

        self._update_runtime_row(
            self.gazebo_runtime,
            self.gazebo_runtime_led,
            self._gazebo_started_at,
        )
        self._update_runtime_row(
            self.controller_runtime,
            self.controller_runtime_led,
            self._controller_started_at,
        )

        gazebo_owned = self.gazebo_process.state() != QProcess.NotRunning
        controller_owned = (
            self.controller_process.state() != QProcess.NotRunning
        )

        self.start_gazebo_button.setEnabled(not gazebo_online)
        self.stop_gazebo_button.setEnabled(gazebo_owned)

        self.start_controller_button.setEnabled(
            ros_online
            and gazebo_online
            and active_controller is None
            and not controller_owned
        )
        self.stop_controller_button.setEnabled(controller_owned)

        if gazebo_online:
            if gazebo_owned:
                self.gazebo_state_label.setText("Gazebo: ONLINE (GUI-owned)")
                self.gazebo_state_label.setStyleSheet(
                    f"QLabel {{ font-weight: 600; color: {COLOR_OK}; }}"
                )
            else:
                self.gazebo_state_label.setText("Gazebo: ONLINE (external)")
                self.gazebo_state_label.setStyleSheet(
                    f"QLabel {{ font-weight: 600; color: {COLOR_WARN}; }}"
                )
        else:
            self.gazebo_state_label.setText("Gazebo: OFFLINE")
            self.gazebo_state_label.setStyleSheet(
                f"QLabel {{ font-weight: 600; color: {COLOR_ERR}; }}"
            )

        self._update_system_chip(ros_online, gazebo_online, active_controller)

    def _update_system_chip(self, ros_online, gazebo_online, active_controller):
        if active_controller is not None:
            self.system_chip.setText("EXPERIMENT RUNNING")
            self.system_chip.setStyleSheet(
                f"QLabel#systemChip {{ background-color: {COLOR_OK}; "
                "color: #10241b; }"
            )
        elif gazebo_online:
            self.system_chip.setText("SIMULATION READY")
            self.system_chip.setStyleSheet(
                f"QLabel#systemChip {{ background-color: {COLOR_WARN}; "
                "color: #2b2110; }"
            )
        else:
            self.system_chip.setText("IDLE")
            self.system_chip.setStyleSheet(
                "QLabel#systemChip { background-color: #3a404a; "
                "color: #c7ced6; }"
            )

    def _update_runtime_row(self, label: QLabel, led: QLabel, started_at):
        if started_at is None:
            label.setText("—")
            label.setStyleSheet(
                f"QLabel {{ color: {COLOR_DIM}; font-weight: 600; }}"
            )
            self._set_led(led, COLOR_DIM)
        else:
            elapsed = int(time.monotonic() - started_at)
            label.setText(f"{elapsed // 60:02d}:{elapsed % 60:02d}")
            label.setStyleSheet(
                f"QLabel {{ color: {COLOR_OK}; font-weight: 600; }}"
            )
            self._set_led(led, COLOR_OK)

    @staticmethod
    def _find_active_controller(nodes: Set[str]) -> Optional[str]:
        for node in CONTROLLER_NODES.values():
            if node in nodes:
                return node
        return None

    def _set_status(self, label: QLabel, led: QLabel, ok: bool, text: str):
        label.setText(text)
        if ok:
            label.setStyleSheet(f"QLabel {{ color: {COLOR_OK}; font-weight: 600; }}")
            self._set_led(led, COLOR_OK)
        else:
            label.setStyleSheet(f"QLabel {{ color: {COLOR_ERR}; font-weight: 600; }}")
            self._set_led(led, COLOR_ERR)

    # ------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------

    def append_console(self, tag: str, message: str):
        entry = (time.strftime("%H:%M:%S"), tag, message)
        self._log_entries.append(entry)
        if len(self._log_entries) > 10000:
            self._log_entries = self._log_entries[-10000:]

        if self._entry_visible(entry):
            self.console.append(self._format_entry(entry))
            self._maybe_autoscroll()

    def _entry_visible(self, entry) -> bool:
        timestamp, tag, message = entry
        if self._active_filter != "ALL" and tag != self._active_filter:
            return False
        needle = self.search_box.text().strip().lower()
        if needle and needle not in message.lower() and needle not in tag.lower():
            return False
        return True

    @staticmethod
    def _format_entry(entry) -> str:
        timestamp, tag, message = entry
        tag_color = TAG_COLORS.get(tag, "#98a1ab")
        return (
            f'<span style="color:#6b7480">{timestamp}</span> '
            f'<span style="color:{tag_color};font-weight:600">[{tag}]</span> '
            f'<span style="color:#c9d1d9">{html.escape(message)}</span>'
        )

    def _console_filter_changed(self, button):
        self._active_filter = button.text()
        self._refresh_console()

    def _refresh_console(self):
        visible = [
            self._format_entry(entry)
            for entry in self._log_entries[-5000:]
            if self._entry_visible(entry)
        ]
        self.console.setHtml(
            "<html><body style='white-space:pre'>"
            + "<br>".join(visible)
            + "</body></html>"
        )
        self._maybe_autoscroll()

    def _maybe_autoscroll(self):
        if self.autoscroll_checkbox.isChecked():
            scrollbar = self.console.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def console_clear(self):
        self._log_entries.clear()
        self.console.clear()

    def console_save(self):
        default_name = "duckiedrone_console_" + time.strftime(
            "%Y%m%d_%H%M%S"
        ) + ".log"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save console log",
            default_name,
            "Log files (*.log);;Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                for timestamp, tag, message in self._log_entries:
                    handle.write(f"{timestamp} [{tag}] {message}\n")
            self.append_console("GUI", f"Console log saved to: {path}")
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Save failed",
                f"Could not save the console log:\n{exc}",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bool_to_ros(value: bool) -> str:
        return "true" if value else "false"

    def closeEvent(self, event):
        owned_running = (
            self.controller_process.state() != QProcess.NotRunning
            or self.gazebo_process.state() != QProcess.NotRunning
        )

        if owned_running:
            answer = QMessageBox.question(
                self,
                "Exit Duckiedrone Control Panel",
                "GUI-owned ROS processes are still running.\n"
                "Stop them and close the GUI?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if answer != QMessageBox.Yes:
                event.ignore()
                return

        self._shutdown_process(self.controller_process, "CONTROLLER")
        self._shutdown_process(self.gazebo_process, "GAZEBO")

        # ----------------------------------------------------
        # Final sweep: kill any orphaned project processes that
        # were NOT started by this GUI (external roslaunch, etc.)
        # ----------------------------------------------------
        sweep_command = (
            f"source '{ROS_SETUP}' && source '{WS_SETUP}' && "
            "rosnode kill /physics_mpc_controller /pem_mpc_controller "
            "/pid_controller /mpc_controller /vstmpc_controller /scenario_runner "
            "/odom_bridge /mixer_node >/dev/null 2>&1 || true; "
            "pkill -f run_scenario.launch >/dev/null 2>&1 || true; "
            "pkill -TERM -x gzclient >/dev/null 2>&1 || true; "
            "pkill -TERM -x gzserver >/dev/null 2>&1 || true"
        )
        try:
            subprocess.run(
                ["/bin/bash", "-lc", sweep_command],
                timeout=3.0,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass

        event.accept()

    def _shutdown_process(self, process: QProcess, tag: str):
        if process.state() == QProcess.NotRunning:
            return

        if tag == "GAZEBO":
            cleanup_command = (
                f"source '{ROS_SETUP}' && "
                f"source '{WS_SETUP}' && "
                "rosnode kill /mixer_node /gazebo_gui /gazebo "
                ">/dev/null 2>&1 || true; "
                "sleep 0.5; "
                "pkill -TERM -x gzclient >/dev/null 2>&1 || true; "
                "pkill -TERM -x gzserver >/dev/null 2>&1 || true"
            )
            try:
                subprocess.run(
                    ["/bin/bash", "-lc", cleanup_command],
                    timeout=3.0,
                    check=False,
                )
            except (subprocess.TimeoutExpired, OSError):
                pass

        pid = int(process.processId())
        try:
            os.kill(pid, signal.SIGINT)
        except OSError:
            process.terminate()

        if not process.waitForFinished(4000):
            self.append_console(
                tag,
                "Sending SIGTERM during GUI shutdown."
            )
            process.terminate()

        if not process.waitForFinished(1500):
            self.append_console(
                tag,
                "Force-killing process during GUI shutdown."
            )
            process.kill()
            process.waitForFinished(1000)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_SHEET)
    if os.path.isfile(ICON_PATH):
        app.setWindowIcon(QIcon(ICON_PATH))
    window = DuckiedroneControlPanel()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()