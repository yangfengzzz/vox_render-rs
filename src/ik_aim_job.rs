/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::simd_math::{SimdFloat4, Float4x4};
use crate::simd_quaternion::SimdQuaternion;

// ozz::animation::IKAimJob rotates a joint so it aims at a target. Joint aim
// direction and up vectors can be different from joint axis. The job computes
// the transformation (rotation) that needs to be applied to the joints such
// that a provided forward vector (in joint local-space) aims at the target
// position (in skeleton model-space). Up vector (in joint local-space) is also
// used to keep the joint oriented in the same direction as the pole vector.
// The job also exposes an offset (in joint local-space) from where the forward
// vector should aim the target.
// Result is unstable if joint-to-target direction is parallel to pole vector,
// or if target is too close to joint position.
pub struct IKAimJob<'a> {
    // Job input.

    // Target position to aim at, in model-space
    pub target: SimdFloat4,

    // Joint forward axis, in joint local-space, to be aimed at target position.
    // This vector shall be normalized, otherwise validation will fail.
    // Default is x axis.
    pub forward: SimdFloat4,

    // Offset position from the joint in local-space, that will aim at target.
    pub offset: SimdFloat4,

    // Joint up axis, in joint local-space, used to keep the joint oriented in the
    // same direction as the pole vector. Default is y axis.
    pub up: SimdFloat4,

    // Pole vector, in model-space. The pole vector defines the direction
    // the up should point to.  Note that IK chain orientation will flip when
    // target vector and the pole vector are aligned/crossing each other. It's
    // caller responsibility to ensure that this doesn't happen.
    pub pole_vector: SimdFloat4,

    // Twist_angle rotates joint around the target vector.
    // Default is 0.
    pub twist_angle: f32,

    // Weight given to the IK correction clamped in range [0,1]. This allows to
    // blend / interpolate from no IK applied (0 weight) to full IK (1).
    pub weight: f32,

    // Joint model-space matrix.
    pub joint: Option<&'a Float4x4>,

    // Job output.

    // Output local-space joint correction quaternion. It needs to be multiplied
    // with joint local-space quaternion.
    pub joint_correction: Option<&'a mut SimdQuaternion>,

    // Optional boolean output value, set to true if target can be reached with IK
    // computations. Target is considered not reachable when target is between
    // joint and offset position.
    pub reached: Option<&'a mut bool>,
}

impl<'a> IKAimJob<'a> {
    // Default constructor, initializes default values.
    pub fn new() -> IKAimJob<'a> {
        return IKAimJob {
            target: SimdFloat4::zero(),
            forward: SimdFloat4::x_axis(),
            offset: SimdFloat4::zero(),
            up: SimdFloat4::y_axis(),
            pole_vector: SimdFloat4::y_axis(),
            twist_angle: 0.0,
            weight: 1.0,
            joint: None,
            joint_correction: None,
            reached: None,
        };
    }

    // Validates job parameters. Returns true for a valid job, or false otherwise:
    // -if output quaternion pointer is nullptr
    pub fn validate(&self) -> bool {
        let mut valid = true;
        valid &= self.joint.is_some();
        valid &= self.joint_correction.is_some();
        valid &= self.forward.is_normalized_est3().are_all_true1();
        return valid;
    }

    // Runs job's execution task.
    // The job is validated before any operation is performed, see validate() for
    // more details.
    // Returns false if *this job is not valid.
    pub fn run(&'a mut self) -> bool {
        todo!()
    }
}