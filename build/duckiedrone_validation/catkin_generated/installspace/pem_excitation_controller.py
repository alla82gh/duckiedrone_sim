#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pem_excitation_controller.py

Closed-loop excitation controller for PEM system identification
of the Duckiedrone DD21.

Architecture
------------
    fixed hover reference
            |
       Cascade PID
            |
       u_PID physical
            +
       u_exc PRBS
            |
       ControllerBase.clamp_u()
            |
       AllocationMixer
            |
       /cmd_motors
            |
          plant

The original PID controller is NOT modified.

Identification input is NOT u_exc itself.
The authoritative input remains the body wrench reconstructed
offline from /cmd_motors using the TRUE plant parameters.

State:
    x = [x, y, z,
         phi, theta, psi,
         vx, vy, vz,
         p, q, r]

Physical controller output:
    u = [T, tau_phi, tau_theta, tau_psi]

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
# PEM Excitation Controller
# ======================================================================

class PEMExcitationController(CascadePID):

    CHANNEL_INDEX = {
        "thrust": 0,
        "roll":   1,
        "pitch":  2,
        "yaw":    3,
    }

    DEFAULT_AMPLITUDE = {
        "thrust": 0.30,    # [N]
        "roll":   0.010,   # [N m]
        "pitch":  0.010,   # [N m]
        "yaw":    0.005,   # [N m]
    }

    def __init__(self):

        # --------------------------------------------------------------
        # Build the already validated cascade PID controller.
        # --------------------------------------------------------------
        super(PEMExcitationController, self).__init__()

        # --------------------------------------------------------------
        # Excitation configuration
        # --------------------------------------------------------------

        self.channel = str(
            rospy.get_param("~channel", "thrust")
        ).lower()

        if self.channel not in self.CHANNEL_INDEX:
            raise ValueError(
                "Invalid PEM excitation channel '%s'. "
                "Use: thrust | roll | pitch | yaw"
                % self.channel
            )

        default_amp = self.DEFAULT_AMPLITUDE[self.channel]

        self.amplitude = float(
            rospy.get_param("~amplitude", default_amp)
        )

        self.dwell = float(
            rospy.get_param("~dwell", 0.10)
        )

        self.pre_hover = float(
            rospy.get_param("~pre_hover", 5.0)
        )

        self.duration = float(
            rospy.get_param("~duration", 15.0)
        )

        self.seed = int(
            rospy.get_param("~seed", 25)
        )

        # --------------------------------------------------------------
        # Expected identification operating point
        # --------------------------------------------------------------

        self.hover_z = float(
            rospy.get_param("~hover_z", 1.5)
        )

        # Conditions required before starting the 5 s pre-excitation
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
        # They only switch OFF the excitation and leave PID active.
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
        # Deterministic PRBS generator
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # 8-bit maximal-length LFSR
        #
        # Polynomial:
        #     x^8 + x^6 + x^5 + x^4 + 1
        #
        # Sequence length:
        #     2^8 - 1 = 255 bits
        # --------------------------------------------------------------

        self._lfsr_state = int(self.seed) & 0xFF

        # LFSR must never start from the all-zero state.
        if self._lfsr_state == 0:
            self._lfsr_state = 1

        self._stable_since = None
        self._excitation_start = None

        self._bit_index = -1
        self._prbs_sign = 0.0

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
        #   2 = PRBS active
        #   3 = recovery / excitation complete
        #   4 = excitation safety-aborted
        # --------------------------------------------------------------

        self.exc_pub = rospy.Publisher(
            "/validation/pem_excitation",
            Float32MultiArray,
            queue_size=10
        )

        rospy.loginfo(
            "============================================================"
        )
        rospy.loginfo(
            " PEM excitation controller initialized"
        )
        rospy.loginfo(
            " channel      : %s", self.channel
        )
        rospy.loginfo(
            " amplitude    : %.6f", self.amplitude
        )
        rospy.loginfo(
            " dwell        : %.3f s", self.dwell
        )
        rospy.loginfo(
            " pre-hover    : %.1f s", self.pre_hover
        )
        rospy.loginfo(
            " duration     : %.1f s", self.duration
        )
        rospy.loginfo(
            " seed         : %d", self.seed
        )
        rospy.loginfo(
            " hover z      : %.3f m", self.hover_z
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
        indices_zero = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]

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

    def _new_prbs_sign(self):
        """
        Return the next sample of an 8-bit maximal-length PRBS.

        Polynomial:
            x^8 + x^6 + x^5 + x^4 + 1

        Period:
            2^8 - 1 = 255 bits
        """

        # Output bit from current LFSR state
        output_bit = self._lfsr_state & 0x01

        # Feedback taps for:
        # x^8 + x^6 + x^5 + x^4 + 1
        feedback = (
            ((self._lfsr_state >> 0) ^
            (self._lfsr_state >> 2) ^
            (self._lfsr_state >> 3) ^
            (self._lfsr_state >> 4))
            & 0x01
        )

        # Right-shift register and insert feedback at MSB
        self._lfsr_state = (
            (self._lfsr_state >> 1)
            | (feedback << 7)
        )

        return 1.0 if output_bit else -1.0

    # ------------------------------------------------------------------

    def _prbs_value(self, t_exc):
        """
        Zero-order-held PRBS.

        A new binary value is selected every self.dwell seconds.
        """

        required_index = int(
            np.floor(t_exc / self.dwell)
        )

        while self._bit_index < required_index:

            self._bit_index += 1
            self._prbs_sign = self._new_prbs_sign()

            rospy.loginfo(
                "PEM PRBS: bit=%d channel=%s value=%+.6f",
                self._bit_index,
                self.channel,
                self.amplitude * self._prbs_sign
            )

        return self.amplitude * self._prbs_sign

    # ------------------------------------------------------------------

    def _publish_diag(
        self,
        phase_code,
        t_stable,
        t_exc,
        u_exc
    ):

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
            "PEM IDENTIFICATION PHASE: %s",
            phase
        )

    # ------------------------------------------------------------------

    def control_law(self, x, x_ref):
        """
        PID stabilization + additive physical-input excitation.

        IMPORTANT:
            ControllerBase applies clamp_u() AFTER this method.
            Therefore the excitation is injected before the common
            actuator/input constraints and before rotor allocation.
        """

        # --------------------------------------------------------------
        # Original validated PID
        # --------------------------------------------------------------

        u_pid = super(
            PEMExcitationController,
            self
        ).control_law(
            x,
            x_ref
        )

        u_exc = np.zeros(4)

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

            hover_ref_ok = self._fixed_hover_reference(
                x_ref
            )

            hover_state_ok = self._state_near_hover(
                x
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
                    "PEM: stable hover detected — "
                    "starting %.1f s pre-excitation window",
                    self.pre_hover
                )

            t_stable = now - self._stable_since

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
            # Start PRBS.
            # ----------------------------------------------------------

            self._excitation_start = now
            self._bit_index = -1
            self._prbs_sign = 0.0

            rospy.loginfo(
                "============================================================"
            )
            rospy.loginfo(
                " PEM PRBS START"
            )
            rospy.loginfo(
                " channel   = %s", self.channel
            )
            rospy.loginfo(
                " amplitude = %.6f", self.amplitude
            )
            rospy.loginfo(
                " dwell     = %.3f s", self.dwell
            )
            rospy.loginfo(
                " duration  = %.1f s", self.duration
            )
            rospy.loginfo(
                "============================================================"
            )

        # --------------------------------------------------------------
        # PRBS phase
        # --------------------------------------------------------------

        t_exc = now - self._excitation_start

        # Safety is checked only after the experiment starts.
        if self._safety_violation(x):

            self._excitation_aborted = True

            rospy.logerr(
                "PEM excitation SAFETY ABORT: "
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
                " PEM PRBS COMPLETE — PID recovery only"
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
        # Current PRBS sample
        # --------------------------------------------------------------

        value = self._prbs_value(
            t_exc
        )

        channel_index = self.CHANNEL_INDEX[
            self.channel
        ]

        u_exc[channel_index] = value

        self._phase_log(
            "PRBS ACTIVE"
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

    PEMExcitationController().spin()
