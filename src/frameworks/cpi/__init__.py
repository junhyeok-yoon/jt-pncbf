"""Certified Policy Iteration (CPI), iteration-0: backup-only certificate regression.

Supervised regression of the single-backup (deadband-brake m_0) stopping value with a conservative
(pinball) loss and split-conformal calibration. No policy training, no filter, no JT. This package may
import src.common / src.envs / src.eval; it must not import jt_pncbf or oc_pncbf (05_code §2).
"""
