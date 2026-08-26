#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pem_mimo_excitation_controller.py

Closed-loop MIMO excitation controller for PEM system identification
of the Duckiedrone DD21.

Architecture
------------
    fixed hover reference
            |
       Cascade PID
            |
       u_PID physical
            +
       u_exc MIMO PRBS
            |
       ControllerBase.clamp_u()
            |
       AllocationMixer
            |
       /cmd_motors
            |
          plant

The original validated PID controller is NOT modified.

Identification input is NOT u_exc itself.
The authoritative input remains the body wrench reconstructed
offline from /cmd_motors using the TRUE plant parameters.

State
-----
x = [
    x, y, z,
    phi, theta, psi,
    vx, vy, vz,
    p, q, r
]

Physical controller output
--------------------------
u = [
    T,
    tau_phi,
    tau_theta,
    tau_psi
]

MIMO excitation
---------------
Four simultaneous PRBS signals are generated with the same
8-bit maximal-length LFSR polynomial used by the validated
SISO excitation controller:

    x^8 + x^6 + x^5 + x^4 + 1

Default amplitudes:
    thrust : +/-0.300 N
    roll   : +/-0.010 N m
    pitch  : +/-0.010 N m
    yaw    : +/-0.005 N m

Default seeds:
    thrust : 42
    roll   : 85
    pitch  : 171
    yaw    : 255

For the first 150 bits these four sequences are:
    - exactly zero mean (75 positive / 75 negative each)
    - full rank as a 150 x 4 input-sequence matrix
    - pairwise correlation magnitude approximately 0.01333

Author: Abdallah GHOUL
2026
"""

import os
import sys

import numpy as np
import rospy
import rospkg

from std_msgs.msg import Float32MultiArray


# ----------------------------------------------------------------------
# Import paths
# ----------------------------------------------------------------------

_PKG = rospkg.RosPack().get_path("duckiedrone_validation")

for _d in ("controllers", "models", "scenarios"):
    _p = os.path.join(_PKG, "scripts", _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pid_controller import CascadePID


# ======================================================================
# PEM MIMO Excitation Controller
# ======================================================================

class PEMMIMOExcitationController(CascadePID):

    CHANNEL_NAMES = (
        "thrust",
        "roll",
        "pitch",
        "yaw",
    )

    DEFAULT_AMPLITUDES = np.array([
        0.300,   # thrust [N]
        0.010,   # roll   [N m]
        0.010,   # pitch  [N m]
        0.005,   # yaw    [N m]
    ], dtype=float)

    DEFAULT_SEEDS = np.array([
        42,
        85,
        171,
        255,
    ], dtype=int)

    def __init__(self):

        # --------------------------------------------------------------
        # Build the already validated cascade PID controller.
        # --------------------------------------------------------------

        super(PEMMIMOExcitationController, self).__init__()

        # --------------------------------------------------------------
        # MIMO excitation amplitudes
        # --------------------------------------------------------------

        self.amplitudes = np.array([
            float(rospy.get_param(
                "~thrust_amplitude",
                self.DEFAULT_AMPLITUDES[0]
            )),
            float(rospy.get_param(
                "~roll_amplitude",
                self.DEFAULT_AMPLITUDES[1]
            )),
            float(rospy.get_param(
                "~pitch_amplitude",
                self.DEFAULT_AMPLITUDES[2]
            )),
            float(rospy.get_param(
                "~yaw_amplitude",
                self.DEFAULT_AMPLITUDES[3]
            )),
        ], dtype=float)

        if np.any(self.amplitudes < 0.0):
            raise ValueError(
                "PEM MIMO excitation amplitudes must be non-negative."
            )

        # --------------------------------------------------------------
        # MIMO PRBS seeds
        # --------------------------------------------------------------

        self.seeds = np.array([
            int(rospy.get_param(
                "~thrust_seed",
                int(self.DEFAULT_SEEDS[0])
            )),
            int(rospy.get_param(
                "~roll_seed",
                int(self.DEFAULT_SEEDS[1])
            )),
            int(rospy.get_param(
                "~pitch_seed",
                int(self.DEFAULT_SEEDS[2])
            )),
            int(rospy.get_param(
                "~yaw_seed",
                int(self.DEFAULT_SEEDS[3])
            )),
        ], dtype=int)

        # Keep the same 8-bit LFSR convention as the SISO controller.
        self._lfsr_states = np.bitwise_and(
            self.seeds,
            0xFF
        ).astype(int)

        # LFSR must never start from the all-zero state.
        self._lfsr_states[
            self._lfsr_states == 0
        ] = 1

        # --------------------------------------------------------------
        # Experiment timing
        # --------------------------------------------------------------

        self.dwell = float(
            rospy.get_param("~dwell", 0.10)
        )

        self.pre_hover = float(
            rospy.get_param("~pre_hover", 5.0)
        )

        self.duration = float(
            rospy.get_param("~duration", 15.0)
        )

        if self.dwell <= 0.0:
            raise ValueError(
                "PEM MIMO dwell must be > 0."
            )

        if self.pre_hover < 0.0:
            raise ValueError(
                "PEM MIMO pre_hover must be >= 0."
            )

        if self.duration <= 0.0:
            raise ValueError(
                "PEM MIMO duration must be > 0."
            )

        # --------------------------------------------------------------
        # Expected identification operating point
        # --------------------------------------------------------------

        self.hover_z = float(
            rospy.get_param("~hover_z", 1.5)
        )

        # Conditions required before starting the pre-excitation
        # steady-hover clock.

        self.ref_z_tol = float(
            rospy.get_param("~ref_z_tol", 0.01)
        )

        self.state_z_tol = float(
            rospy.get_param("~state_z_tol", 0.10)
        )

        self.vz_tol = float(
            rospy.get_param("~vz_tol", 0.15)
        )

        self.angle_tol = float(
            rospy.get_param("~angle_tol", 0.10)
        )

        # --------------------------------------------------------------
        # Safety abort limits
        #
        # These do NOT replace ControllerBase hard input limits.
        # They only switch OFF excitation and leave PID active.
        # --------------------------------------------------------------

        self.safety_angle = float(
            rospy.get_param("~safety_angle", 0.20)
        )

        self.safety_z_error = float(
            rospy.get_param("~safety_z_error", 0.40)
        )

        self.safety_vz = float(
            rospy.get_param("~safety_vz", 1.00)
        )

        # --------------------------------------------------------------
        # Experiment state
        # --------------------------------------------------------------

        self._stable_since = None
        self._excitation_start = None

        # One common bit clock; four independent LFSR states.
        self._bit_index = -1
        self._prbs_signs = np.zeros(
            4,
            dtype=float
        )

        self._excitation_finished = False
        self._excitation_aborted = False

        self._last_phase = None

        # --------------------------------------------------------------
        # Diagnostic topic
        #
        # data =
        # [
        #   phase_code,
        #   t_stable,
        #   t_exc,
        #   dT_exc,
        #   tau_phi_exc,
        #   tau_theta_exc,
        #   tau_psi_exc
        # ]
        #
        # phase_code:
        #   0 = waiting for stable hover
        #   1 = pre-excitation steady hover
        #   2 = MIMO PRBS active
        #   3 = recovery / excitation complete
        #   4 = excitation safety-aborted
        # --------------------------------------------------------------

        self.exc_pub = rospy.Publisher(
            "/validation/pem_excitation",
            Float32MultiArray,
            queue_size=10
        )

        # --------------------------------------------------------------
        # Initialization log
        # --------------------------------------------------------------

        rospy.loginfo(
            "============================================================"
        )
        rospy.loginfo(
            " PEM MIMO excitation controller initialized"
        )
        rospy.loginfo(
            " amplitudes   : "
            "dT=%+.6f N  roll=%+.6f  pitch=%+.6f  yaw=%+.6f N m",
            self.amplitudes[0],
            self.amplitudes[1],
            self.amplitudes[2],
            self.amplitudes[3]
        )
        rospy.loginfo(
            " seeds        : "
            "thrust=%d roll=%d pitch=%d yaw=%d",
            self._lfsr_states[0],
            self._lfsr_states[1],
            self._lfsr_states[2],
            self._lfsr_states[3]
        )
        rospy.loginfo(
            " dwell        : %.3f s",
            self.dwell
        )
        rospy.loginfo(
            " pre-hover    : %.1f s",
            self.pre_hover
        )
        rospy.loginfo(
            " duration     : %.1f s",
            self.duration
        )
        rospy.loginfo(
            " hover z      : %.3f m",
            self.hover_z
        )
        rospy.loginfo(
            "============================================================"
        )

    # ------------------------------------------------------------------

    def _fixed_hover_reference(self, x_ref):
        """
        Return True only when the scenario reference has reached the
        fixed S1 hover operating point.

        This prevents the PRBS clock from starting during the 4 s
        takeoff ramp.
        """

        if abs(x_ref[2] - self.hover_z) > self.ref_z_tol:
            return False

        # S1 must remain a fixed hover reference.
        indices_zero = [
            0, 1,
            3, 4, 5,
            6, 7, 8,
            9, 10, 11
        ]

        for i in indices_zero:
            if abs(x_ref[i]) > 1.0e-6:
                return False

        return True

    # ------------------------------------------------------------------

    def _state_near_hover(self, x):
        """
        Conservative condition used only to start the identification
        experiment.

        It is NOT used as an identification model assumption.
        """

        if abs(x[2] - self.hover_z) > self.state_z_tol:
            return False

        if abs(x[8]) > self.vz_tol:
            return False

        if abs(x[3]) > self.angle_tol:
            return False

        if abs(x[4]) > self.angle_tol:
            return False

        return True

    # ------------------------------------------------------------------

    def _safety_violation(self, x):
        """
        If the aircraft leaves the intended local identification region,
        stop excitation immediately and let the PID recover.
        """

        if abs(x[3]) > self.safety_angle:
            return True

        if abs(x[4]) > self.safety_angle:
            return True

        if abs(x[2] - self.hover_z) > self.safety_z_error:
            return True

        if abs(x[8]) > self.safety_vz:
            return True

        return False

    # ------------------------------------------------------------------

    @staticmethod
    def _advance_lfsr(state):
        """
        Advance one 8-bit maximal-length LFSR by one sample.

        Polynomial:
            x^8 + x^6 + x^5 + x^4 + 1

        Period:
            2^8 - 1 = 255 bits

        Returns
        -------
        new_state : int
        sign      : float
            +1.0 or -1.0
        """

        state = int(state) & 0xFF

        if state == 0:
            state = 1

        output_bit = state & 0x01

        feedback = (
            ((state >> 0) ^
             (state >> 2) ^
             (state >> 3) ^
             (state >> 4))
            & 0x01
        )

        new_state = (
            (state >> 1)
            | (feedback << 7)
        )

        sign = (
            1.0
            if output_bit
            else -1.0
        )

        return new_state, sign

    # ------------------------------------------------------------------

    def _new_prbs_vector(self):
        """
        Advance all four independent LFSRs by one common bit clock.

        Returns
        -------
        signs : ndarray, shape (4,)
            [s_T, s_phi, s_theta, s_psi]
        """

        signs = np.zeros(
            4,
            dtype=float
        )

        for i in range(4):

            new_state, sign = self._advance_lfsr(
                self._lfsr_states[i]
            )

            self._lfsr_states[i] = new_state
            signs[i] = sign

        return signs

    # ------------------------------------------------------------------

    def _prbs_vector(self, t_exc):
        """
        Zero-order-held four-channel MIMO PRBS.

        A new four-component binary vector is selected every
        self.dwell seconds.
        """

        required_index = int(
            np.floor(t_exc / self.dwell)
        )

        while self._bit_index < required_index:

            self._bit_index += 1

            self._prbs_signs = (
                self._new_prbs_vector()
            )

            u_exc = (
                self.amplitudes
                * self._prbs_signs
            )

            rospy.loginfo(
                "PEM MIMO PRBS: bit=%d "
                "dT=%+.6f "
                "roll=%+.6f "
                "pitch=%+.6f "
                "yaw=%+.6f",
                self._bit_index,
                u_exc[0],
                u_exc[1],
                u_exc[2],
                u_exc[3]
            )

        return (
            self.amplitudes
            * self._prbs_signs
        )

    # ------------------------------------------------------------------

    def _publish_diag(
        self,
        phase_code,
        t_stable,
        t_exc,
        u_exc
    ):
        """
        Publish the same 7-value diagnostic interface used by the
        SISO PEM excitation controller, so the validated collector
        remains unchanged.
        """

        msg = Float32MultiArray()

        msg.data = [
            float(phase_code),
            float(t_stable),
            float(t_exc),
            float(u_exc[0]),
            float(u_exc[1]),
            float(u_exc[2]),
            float(u_exc[3]),
        ]

        self.exc_pub.publish(msg)

    # ------------------------------------------------------------------

    def _phase_log(self, phase):
        """
        Print each experiment phase only once.
        """

        if phase == self._last_phase:
            return

        self._last_phase = phase

        rospy.loginfo(
            "PEM MIMO IDENTIFICATION PHASE: %s",
            phase
        )

    # ------------------------------------------------------------------

    def control_law(self, x, x_ref):
        """
        PID stabilization + additive four-channel physical-input
        excitation.

        IMPORTANT
        ---------
        ControllerBase applies clamp_u() AFTER this method.
        Therefore the excitation is injected before the common
        actuator/input constraints and before rotor allocation.
        """

        # --------------------------------------------------------------
        # Original validated PID
        # --------------------------------------------------------------

        u_pid = super(
            PEMMIMOExcitationController,
            self
        ).control_law(
            x,
            x_ref
        )

        u_exc = np.zeros(
            4,
            dtype=float
        )

        now = rospy.get_time()

        # --------------------------------------------------------------
        # Excitation already aborted
        # --------------------------------------------------------------

        if self._excitation_aborted:

            self._phase_log(
                "SAFETY ABORT — PID RECOVERY"
            )

            self._publish_diag(
                4,
                0.0,
                0.0,
                u_exc
            )

            return u_pid

        # --------------------------------------------------------------
        # Excitation already completed
        # --------------------------------------------------------------

        if self._excitation_finished:

            self._phase_log(
                "RECOVERY — PID ONLY"
            )

            self._publish_diag(
                3,
                0.0,
                self.duration,
                u_exc
            )

            return u_pid

        # --------------------------------------------------------------
        # Before excitation starts:
        # require true stable S1 hover.
        # --------------------------------------------------------------

        if self._excitation_start is None:

            hover_ref_ok = (
                self._fixed_hover_reference(
                    x_ref
                )
            )

            hover_state_ok = (
                self._state_near_hover(
                    x
                )
            )

            if not (
                hover_ref_ok
                and hover_state_ok
            ):

                self._stable_since = None

                self._phase_log(
                    "WAITING FOR STABLE HOVER"
                )

                self._publish_diag(
                    0,
                    0.0,
                    0.0,
                    u_exc
                )

                return u_pid

            # ----------------------------------------------------------
            # Stable hover detected.
            # Start / continue pre-excitation timer.
            # ----------------------------------------------------------

            if self._stable_since is None:

                self._stable_since = now

                rospy.loginfo(
                    "PEM MIMO: stable hover detected — "
                    "starting %.1f s pre-excitation window",
                    self.pre_hover
                )

            t_stable = (
                now
                - self._stable_since
            )

            if t_stable < self.pre_hover:

                self._phase_log(
                    "PRE-EXCITATION STEADY HOVER"
                )

                self._publish_diag(
                    1,
                    t_stable,
                    0.0,
                    u_exc
                )

                return u_pid

            # ----------------------------------------------------------
            # Start MIMO PRBS.
            # ----------------------------------------------------------

            self._excitation_start = now
            self._bit_index = -1
            self._prbs_signs[:] = 0.0

            rospy.loginfo(
                "============================================================"
            )
            rospy.loginfo(
                " PEM MIMO PRBS START"
            )
            rospy.loginfo(
                " amplitudes : "
                "dT=%+.6f roll=%+.6f pitch=%+.6f yaw=%+.6f",
                self.amplitudes[0],
                self.amplitudes[1],
                self.amplitudes[2],
                self.amplitudes[3]
            )
            rospy.loginfo(
                " seeds      : "
                "thrust=%d roll=%d pitch=%d yaw=%d",
                self.seeds[0],
                self.seeds[1],
                self.seeds[2],
                self.seeds[3]
            )
            rospy.loginfo(
                " dwell      : %.3f s",
                self.dwell
            )
            rospy.loginfo(
                " duration   : %.1f s",
                self.duration
            )
            rospy.loginfo(
                "============================================================"
            )

        # --------------------------------------------------------------
        # MIMO PRBS phase
        # --------------------------------------------------------------

        t_exc = (
            now
            - self._excitation_start
        )

        # Safety is checked only after the experiment starts.
        if self._safety_violation(x):

            self._excitation_aborted = True

            rospy.logerr(
                "PEM MIMO excitation SAFETY ABORT: "
                "z=%.3f phi=%.4f theta=%.4f vz=%.3f",
                x[2],
                x[3],
                x[4],
                x[8]
            )

            self._publish_diag(
                4,
                self.pre_hover,
                t_exc,
                u_exc
            )

            return u_pid

        # --------------------------------------------------------------
        # End excitation
        # --------------------------------------------------------------

        if t_exc >= self.duration:

            self._excitation_finished = True

            rospy.loginfo(
                "============================================================"
            )
            rospy.loginfo(
                " PEM MIMO PRBS COMPLETE — PID recovery only"
            )
            rospy.loginfo(
                "============================================================"
            )

            self._publish_diag(
                3,
                self.pre_hover,
                t_exc,
                u_exc
            )

            return u_pid

        # --------------------------------------------------------------
        # Current MIMO PRBS vector
        # --------------------------------------------------------------

        u_exc = self._prbs_vector(
            t_exc
        )

        self._phase_log(
            "MIMO PRBS ACTIVE"
        )

        self._publish_diag(
            2,
            self.pre_hover,
            t_exc,
            u_exc
        )

        # --------------------------------------------------------------
        # Physical input injection.
        #
        # clamp_u() and rotor allocation are performed later by
        # ControllerBase.
        # --------------------------------------------------------------

        return u_pid + u_exc


# ======================================================================

if __name__ == "__main__":

    PEMMIMOExcitationController().spin()