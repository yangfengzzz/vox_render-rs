/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::simd_quaternion::SimdQuaternion;
use crate::simd_math::{Float4x4, SimdFloat4};

// ozz::animation::IKTwoBoneJob performs inverse kinematic on a three joints
// chain (two bones).
// The job computes the transformations (rotations) that needs to be applied to
// the first two joints of the chain (named start and middle joints) such that
// the third joint (named end) reaches the provided target position (if
// possible). The job outputs start and middle joint rotation corrections as
// quaternions.
// The three joints must be ancestors, but don't need to be direct
// ancestors (joints in-between will simply remain fixed).
// Implementation is inspired by Autodesk Maya 2 bone IK, improved stability
// wise and extended with Soften IK.
pub struct IKTwoBoneJob<'a> {
    // Job input.

    // Target IK position, in model-space. This is the position the end of the
    // joint chain will try to reach.
    pub target: SimdFloat4,

    // Normalized middle joint rotation axis, in middle joint local-space. Default
    // value is z axis. This axis is usually fixed for a given skeleton (as it's
    // in middle joint space). Its direction is defined like this: a positive
    // rotation around this axis will open the angle between the two bones. This
    // in turn also to define which side the two joints must bend. Job validation
    // will fail if mid_axis isn't normalized.
    pub mid_axis: SimdFloat4,

    // Pole vector, in model-space. The pole vector defines the direction the
    // middle joint should point to, allowing to control IK chain orientation.
    // Note that IK chain orientation will flip when target vector and the pole
    // vector are aligned/crossing each other. It's caller responsibility to
    // ensure that this doesn't happen.
    pub pole_vector: SimdFloat4,

    // Twist_angle rotates IK chain around the vector define by start-to-target
    // vector. Default is 0.
    pub twist_angle: f32,

    // Soften ratio allows the chain to gradually fall behind the target
    // position. This prevents the joint chain from snapping into the final
    // position, softening the final degrees before the joint chain becomes flat.
    // This ratio represents the distance to the end, from which softening is
    // starting.
    pub soften: f32,

    // Weight given to the IK correction clamped in range [0,1]. This allows to
    // blend / interpolate from no IK applied (0 weight) to full IK (1).
    pub weight: f32,

    // Model-space matrices of the start, middle and end joints of the chain.
    // The 3 joints should be ancestors. They don't need to be direct
    // ancestors though.
    pub start_joint: Option<&'a Float4x4>,
    pub mid_joint: Option<&'a Float4x4>,
    pub end_joint: Option<&'a Float4x4>,

    // Job output.

    // Local-space corrections to apply to start and middle joints in order for
    // end joint to reach target position.
    // These quaternions must be multiplied to the local-space quaternion of their
    // respective joints.
    pub start_joint_correction: Option<&'a mut SimdQuaternion>,
    pub mid_joint_correction: Option<&'a mut SimdQuaternion>,

    // Optional boolean output value, set to true if target can be reached with IK
    // computations. Reachability is driven by bone chain length, soften ratio and
    // target distance. Target is considered unreached if weight is less than 1.
    pub reached: Option<&'a mut bool>,
}

impl<'a> IKTwoBoneJob<'a> {
    // Constructor, initializes default values.
    pub fn new() -> IKTwoBoneJob<'a> {
        return IKTwoBoneJob {
            target: SimdFloat4::zero(),
            mid_axis: SimdFloat4::z_axis(),
            pole_vector: SimdFloat4::y_axis(),
            twist_angle: 0.0,
            soften: 0.0,
            weight: 0.0,
            start_joint: None,
            mid_joint: None,
            end_joint: None,
            start_joint_correction: None,
            mid_joint_correction: None,
            reached: None,
        };
    }

    // Validates job parameters. Returns true for a valid job, or false otherwise:
    // -if any input pointer is nullptr
    // -if mid_axis isn't normalized.
    pub fn validate(&self) -> bool {
        let mut valid = true;
        valid &= self.start_joint.is_some() && self.mid_joint.is_some() && self.end_joint.is_some();
        valid &= self.start_joint_correction.is_some() && self.mid_joint_correction.is_some();
        valid &= self.mid_axis.is_normalized_est3().are_all_true1();
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