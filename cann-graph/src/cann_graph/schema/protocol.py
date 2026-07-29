"""Protocol specification models for CANN experiment."""

from typing import Literal

from pydantic import BaseModel, Field

from cann_graph.schema.enums import ProtocolType


class ProtocolSpec(BaseModel):
    """Specification for an experimental protocol in the CANN model."""

    id: str = Field(..., description="Unique identifier for this protocol")
    display_name: str = Field(..., description="Human-readable name")
    kinematics_type: ProtocolType = Field(
        ..., description="Type of kinematic deformation"
    )
    input_name: str = Field(
        default="x", description="Name of the scalar input deformation variable"
    )
    target_stress_name: str = Field(
        default="stress_target", description="Name of the target stress component"
    )
    stress_component: str = Field(
        ..., description="Stress component being measured (e.g., 'P11', 'P22')"
    )
    loss_weight_mode: Literal["manual", "inverse_rms_squared"] = Field(
        default="inverse_rms_squared",
        description="Mode for computing protocol loss weight"
    )
    manual_loss_weight: float | None = Field(
        default=None, description="Manual loss weight (if mode is 'manual')"
    )
    enabled: bool = Field(default=True, description="Whether this protocol is active")

    model_config = {
        "json_schema_extra": {
            "description": "Protocol specification with kinematics and loss configuration"
        }
    }


# Predefined protocol specifications for the nine-protocol thesis model
DEFAULT_PROTOCOLS = [
    ProtocolSpec(
        id="utc_fiber",
        display_name="Uniaxial Tension - Fiber",
        kinematics_type=ProtocolType.UTC_FIBER,
        stress_component="P_ff",
    ),
    ProtocolSpec(
        id="utc_sheet",
        display_name="Uniaxial Tension - Sheet",
        kinematics_type=ProtocolType.UTC_SHEET,
        stress_component="P_ss",
    ),
    ProtocolSpec(
        id="utc_normal",
        display_name="Uniaxial Tension - Normal",
        kinematics_type=ProtocolType.UTC_NORMAL,
        stress_component="P_nn",
    ),
    ProtocolSpec(
        id="ss_fiber",
        display_name="Simple Shear - Fiber",
        kinematics_type=ProtocolType.SS_FIBER,
        stress_component="P_fs",
    ),
    ProtocolSpec(
        id="ss_sheet",
        display_name="Simple Shear - Sheet",
        kinematics_type=ProtocolType.SS_SHEET,
        stress_component="P_sf",
    ),
    ProtocolSpec(
        id="ss_normal",
        display_name="Simple Shear - Normal",
        kinematics_type=ProtocolType.SS_NORMAL,
        stress_component="P_sn",
    ),
    ProtocolSpec(
        id="fs_fs",
        display_name="Finite Strain - Fiber/Sheet",
        kinematics_type=ProtocolType.FS_FS,
        stress_component="P_ff",
    ),
    ProtocolSpec(
        id="fs_fn",
        display_name="Finite Strain - Fiber/Normal",
        kinematics_type=ProtocolType.FS_FN,
        stress_component="P_ff",
    ),
    ProtocolSpec(
        id="fs_sn",
        display_name="Finite Strain - Sheet/Normal",
        kinematics_type=ProtocolType.FS_SN,
        stress_component="P_ss",
    ),
]
