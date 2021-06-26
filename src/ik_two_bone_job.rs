/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::simd_quaternion::SimdQuaternion;
use crate::simd_math::*;

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
        if !self.validate() {
            return false;
        }

        // Early out if weight is 0.
        if self.weight <= 0.0 {
            // No correction.
            **self.start_joint_correction.as_mut().unwrap() = SimdQuaternion::identity();
            **self.mid_joint_correction.as_mut().unwrap() = SimdQuaternion::identity();
            // Target isn't reached.
            if self.reached.is_some() {
                **self.reached.as_mut().unwrap() = false;
            }
            return true;
        }

        // Prepares constant ik data.
        let setup = IKConstantSetup::new(self);

        // Finds soften target position.
        let mut start_target_ss = SimdFloat4::zero();
        let mut start_target_ss_len2 = SimdFloat4::zero();
        let lreached = soften_target(self, &setup,
                                     &mut start_target_ss, &mut start_target_ss_len2);
        if self.reached.is_some() {
            **self.reached.as_mut().unwrap() = lreached && self.weight >= 1.0;
        }

        // Calculate mid_rot_local quaternion which solves for the mid_ss joint
        // rotation.
        let mid_rot_ms = compute_mid_joint(self, &setup, start_target_ss_len2);

        // Calculates end_to_target_rot_ss quaternion which solves for effector
        // rotating onto the target.
        let start_rot_ss = compute_start_joint(self, &setup, &mid_rot_ms,
                                               start_target_ss, start_target_ss_len2);

        // Finally apply weight and output quaternions.
        weight_output(self, &setup, &start_rot_ss, &mid_rot_ms);

        return true;
    }
}

// Local data structure used to share constant data accross ik stages.
struct IKConstantSetup {
    // Constants
    one: SimdFloat4,
    m_one: SimdFloat4,
    mask_sign: SimdInt4,

    // Inverse matrices
    inv_start_joint: Float4x4,

    // Bones vectors and length in mid and start spaces (_ms and _ss).
    start_mid_ms: SimdFloat4,
    mid_end_ms: SimdFloat4,
    start_mid_ss: SimdFloat4,
    start_mid_ss_len2: SimdFloat4,
    mid_end_ss_len2: SimdFloat4,
    start_end_ss_len2: SimdFloat4,
}

impl IKConstantSetup {
    fn new(_job: &IKTwoBoneJob) -> IKConstantSetup {
        let mut setup = IKConstantSetup {
            one: SimdFloat4::zero(),
            m_one: SimdFloat4::zero(),
            mask_sign: SimdInt4::zero(),
            inv_start_joint: Float4x4::identity(),
            start_mid_ms: SimdFloat4::zero(),
            mid_end_ms: SimdFloat4::zero(),
            start_mid_ss: SimdFloat4::zero(),
            start_mid_ss_len2: SimdFloat4::zero(),
            mid_end_ss_len2: SimdFloat4::zero(),
            start_end_ss_len2: SimdFloat4::zero(),
        };

        // Prepares constants
        setup.one = SimdFloat4::one();
        setup.mask_sign = SimdInt4::mask_sign();
        setup.m_one = setup.one.xor_fi(setup.mask_sign);

        // Computes inverse matrices required to change to start and mid spaces.
        // If matrices aren't invertible, they'll be all 0 (ozz::math
        // implementation), which will result in identity correction quaternions.
        let mut invertible = SimdInt4::zero();
        setup.inv_start_joint = _job.start_joint.unwrap().invert(Some(&mut invertible));
        let inv_mid_joint = _job.mid_joint.unwrap().invert(Some(&mut invertible));

        // Transform some positions to mid joint space (_ms)
        let start_ms = inv_mid_joint.transform_point(_job.start_joint.unwrap().cols[3]);
        let end_ms = inv_mid_joint.transform_point(_job.end_joint.unwrap().cols[3]);

        // Transform some positions to start joint space (_ss)
        let mid_ss = setup.inv_start_joint.transform_point(_job.mid_joint.unwrap().cols[3]);
        let end_ss = setup.inv_start_joint.transform_point(_job.end_joint.unwrap().cols[3]);

        // Computes bones vectors and length in mid and start spaces.
        // Start joint position will be treated as 0 because all joints are
        // expressed in start joint space.
        setup.start_mid_ms = -start_ms;
        setup.mid_end_ms = end_ms;
        setup.start_mid_ss = mid_ss;
        let mid_end_ss = end_ss - mid_ss;
        let start_end_ss = end_ss;
        setup.start_mid_ss_len2 = setup.start_mid_ss.length3sqr();
        setup.mid_end_ss_len2 = mid_end_ss.length3sqr();
        setup.start_end_ss_len2 = start_end_ss.length3sqr();
        return setup;
    }
}

// Smoothen target position when it's further that a ratio of the joint chain
// length, and start to target length isn't 0.
// Inspired by http://www.softimageblog.com/archives/108
// and http://www.ryanjuckett.com/programming/analytic-two-bone-ik-in-2d/
fn soften_target(_job: &IKTwoBoneJob, _setup: &IKConstantSetup,
                 _start_target_ss: &mut SimdFloat4,
                 _start_target_ss_len2: &mut SimdFloat4) -> bool {
    // Hanlde position in start joint space (_ss)
    let start_target_original_ss = _setup.inv_start_joint.transform_point(_job.target);
    let start_target_original_ss_len2 = start_target_original_ss.length3sqr();
    let lengths = _setup.start_mid_ss_len2.set_y(_setup.mid_end_ss_len2).set_z(start_target_original_ss_len2).sqrt();
    let start_mid_ss_len = lengths;
    let mid_end_ss_len = lengths.splat_y();
    let start_target_original_ss_len = lengths.splat_z();
    let bone_len_diff_abs = (start_mid_ss_len - mid_end_ss_len).and_not(_setup.mask_sign);
    let bones_chain_len = start_mid_ss_len + mid_end_ss_len;
    // da.yzw needs to be 0
    let da = bones_chain_len * SimdFloat4::zero().clamp(SimdFloat4::load_x(_job.soften), _setup.one);
    let ds = bones_chain_len - da;

    // Sotftens target position if it is further than a ratio (_soften) of the
    // whole bone chain length. Needs to check also that ds and
    // start_target_original_ss_len2 are != 0, because they're used as a
    // denominator.
    // x = start_target_original_ss_len > da
    // y = start_target_original_ss_len > 0
    // z = start_target_original_ss_len > bone_len_diff_abs
    // w = ds                           > 0
    let left = start_target_original_ss_len.set_w(ds);
    let right = da.set_z(bone_len_diff_abs);
    let comp = left.cmp_gt(right);
    let comp_mask = comp.move_mask();

    // xyw all 1, z is untested.
    if (comp_mask & 0xb) == 0xb {
        // Finds interpolation ratio (aka alpha).
        let alpha = (start_target_original_ss_len - da) * ds.rcp_est_x();
        // Approximate an exponential function with : 1-(3^4)/(alpha+3)^4
        // The derivative must be 1 for x = 0, and y must never exceeds 1.
        // Negative x aren't used.
        let three = SimdFloat4::load1(3.0);
        let op = three.set_y(alpha + three);
        let op2 = op * op;
        let op4 = op2 * op2;
        let ratio = op4 * op4.splat_y().rcp_est_x();

        // Recomputes start_target_ss vector and length.
        let start_target_ss_len = da + ds - ds * ratio;
        *_start_target_ss_len2 = start_target_ss_len * start_target_ss_len;
        *_start_target_ss = start_target_original_ss * (start_target_ss_len * start_target_original_ss_len.rcp_est_x()).splat_x();
    } else {
        *_start_target_ss = start_target_original_ss;
        *_start_target_ss_len2 = start_target_original_ss_len2;
    }

    // The maximum distance we can reach is the soften bone chain length: da
    // (stored in !x). The minimum distance we can reach is the absolute value of
    // the difference of the 2 bone lengths, |d1−d2| (stored in z). x is 0 and z
    // is 1, yw are untested.
    return (comp_mask & 0x5) == 0x4;
}

fn compute_mid_joint(_job: &IKTwoBoneJob, _setup: &IKConstantSetup, _start_target_ss_len2: SimdFloat4) -> SimdQuaternion {
    // Computes expected angle at mid_ss joint, using law of cosine (generalized
    // Pythagorean).
    // c^2 = a^2 + b^2 - 2ab cosC
    // cosC = (a^2 + b^2 - c^2) / 2ab
    // Computes both corrected and initial mid joint angles
    // cosine within a single SimdFloat4 (corrected is x component, initial is y).
    let start_mid_end_sum_ss_len2 = _setup.start_mid_ss_len2 + _setup.mid_end_ss_len2;
    let start_mid_end_ss_half_rlen = (SimdFloat4::load1(0.5) * (_setup.start_mid_ss_len2 * _setup.mid_end_ss_len2).rsqrt_est_xnr()).splat_x();
    // Cos value needs to be clamped, as it will exit expected range if
    // start_target_ss_len2 is longer than the triangle can be (start_mid_ss +
    // mid_end_ss).
    let mid_cos_angles_unclamped =
        (start_mid_end_sum_ss_len2.splat_x() - _start_target_ss_len2.set_y(_setup.start_end_ss_len2)) * start_mid_end_ss_half_rlen;
    let mid_cos_angles = mid_cos_angles_unclamped.clamp(_setup.m_one, _setup.one);

    // Computes corrected angle
    let mid_corrected_angle = mid_cos_angles.acos_x();

    // Computes initial angle.
    // The sign of this angle needs to be decided. It's considered negative if
    // mid-to-end joint is bent backward (mid_axis direction dictates valid
    // bent direction).
    let bent_side_ref = _setup.start_mid_ms.cross3(_job.mid_axis);
    let bent_side_flip = bent_side_ref.dot3(_setup.mid_end_ms).cmp_lt(SimdFloat4::zero()).splat_x();
    let mid_initial_angle = mid_cos_angles.splat_y().acos_x().xor_fi(bent_side_flip.and(_setup.mask_sign));

    // Finally deduces initial to corrected angle difference.
    let mid_angles_diff = mid_corrected_angle - mid_initial_angle;

    // Builds quaternion.
    return SimdQuaternion::from_axis_angle(_job.mid_axis, mid_angles_diff);
}

fn compute_start_joint(_job: &IKTwoBoneJob, _setup: &IKConstantSetup, _mid_rot_ms: &SimdQuaternion,
                       _start_target_ss: SimdFloat4, _start_target_ss_len2: SimdFloat4) -> SimdQuaternion {
    // Pole vector in start joint space (_ss)
    let pole_ss = _setup.inv_start_joint.transform_vector(_job.pole_vector);

    // start_mid_ss with quaternion mid_rot_ms applied.
    let mid_end_ss_final = _setup.inv_start_joint
        .transform_vector(_job.mid_joint.unwrap()
            .transform_vector(_mid_rot_ms
                .transform_vector(_setup.mid_end_ms)));
    let start_end_ss_final = _setup.start_mid_ss + mid_end_ss_final;

    // Quaternion for rotating the effector onto the target
    let end_to_target_rot_ss = SimdQuaternion::from_vectors(start_end_ss_final, _start_target_ss);

    // Calculates rotate_plane_ss quaternion which aligns joint chain plane to
    // the reference plane (pole vector). This can only be computed if start
    // target axis is valid (not 0 length)
    // -------------------------------------------------
    let mut start_rot_ss = end_to_target_rot_ss;
    if _start_target_ss_len2.cmp_gt(SimdFloat4::zero()).are_all_true1() {
        // Computes each plane normal.
        let ref_plane_normal_ss = _start_target_ss.cross3(pole_ss);
        let ref_plane_normal_ss_len2 = ref_plane_normal_ss.length3sqr();
        // Computes joint chain plane normal, which is the same as mid joint axis
        // (same triangle).
        let mid_axis_ss = _setup.inv_start_joint.transform_vector(_job.mid_joint.unwrap().transform_vector(_job.mid_axis));
        let joint_plane_normal_ss = end_to_target_rot_ss.transform_vector(mid_axis_ss);
        let joint_plane_normal_ss_len2 = joint_plane_normal_ss.length3sqr();
        // Computes all reciprocal square roots at once.
        let rsqrts = _start_target_ss_len2.set_y(ref_plane_normal_ss_len2).set_z(joint_plane_normal_ss_len2).rsqrt_est_nr();

        // Computes angle cosine between the 2 normalized normals.
        let rotate_plane_cos_angle = (ref_plane_normal_ss * rsqrts.splat_y()).dot3(joint_plane_normal_ss * rsqrts.splat_z());

        // Computes rotation axis, which is either start_target_ss or
        // -start_target_ss depending on rotation direction.
        let rotate_plane_axis_ss = _start_target_ss * rsqrts.splat_x();
        let start_axis_flip = joint_plane_normal_ss.dot3(pole_ss).splat_x().and_fi(_setup.mask_sign);
        let rotate_plane_axis_flipped_ss = rotate_plane_axis_ss.xor_ff(start_axis_flip);

        // Builds quaternion along rotation axis.
        let rotate_plane_ss = SimdQuaternion::from_axis_cos_angle(
            rotate_plane_axis_flipped_ss, rotate_plane_cos_angle.clamp(_setup.m_one, _setup.one));

        if _job.twist_angle != 0.0 {
            // If a twist angle is provided, rotation angle is rotated along
            // rotation plane axis.
            let twist_ss = SimdQuaternion::from_axis_angle(
                rotate_plane_axis_ss, SimdFloat4::load1(_job.twist_angle));
            start_rot_ss = twist_ss * rotate_plane_ss * end_to_target_rot_ss;
        } else {
            start_rot_ss = rotate_plane_ss * end_to_target_rot_ss;
        }
    }
    return start_rot_ss;
}

fn weight_output(_job: &mut IKTwoBoneJob, _setup: &IKConstantSetup, _start_rot: &SimdQuaternion, _mid_rot: &SimdQuaternion) {
    let zero = SimdFloat4::zero();

    // Fix up quaternions so w is always positive, which is required for NLerp
    // (with identity quaternion) to lerp the shortest path.
    let start_rot_fu = _start_rot.xyzw.xor_fi(_setup.mask_sign.and(_start_rot.xyzw.splat_w().cmp_lt(zero)));
    let mid_rot_fu = _mid_rot.xyzw.xor_fi(_setup.mask_sign.and(_mid_rot.xyzw.splat_w().cmp_lt(zero)));

    if _job.weight < 1.0 {
        // NLerp start and mid joint rotations.
        let identity = SimdFloat4::w_axis();
        let simd_weight = zero.max(SimdFloat4::load1(_job.weight));

        // Lerp
        let start_lerp = identity.lerp(start_rot_fu, simd_weight);
        let mid_lerp = identity.lerp(mid_rot_fu, simd_weight);

        // Normalize
        let rsqrts = start_lerp.length4sqr().set_y(mid_lerp.length4sqr()).rsqrt_est_nr();
        _job.start_joint_correction.as_mut().unwrap().xyzw = start_lerp * rsqrts.splat_x();
        _job.mid_joint_correction.as_mut().unwrap().xyzw = mid_lerp * rsqrts.splat_y();
    } else {
        // Quaternions don't need interpolation
        _job.start_joint_correction.as_mut().unwrap().xyzw = start_rot_fu;
        _job.mid_joint_correction.as_mut().unwrap().xyzw = mid_rot_fu;
    }
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod ik_two_bone_job {
    use crate::math_test_helper::*;
    use crate::simd_math::*;
    use crate::*;
    use crate::ik_two_bone_job::IKTwoBoneJob;

    fn _expect_reached(_job: &IKTwoBoneJob, _reachable: bool) {
        // Computes local transforms
        let mid_local = _job.start_joint.unwrap().invert(None) * *_job.mid_joint.unwrap();
        let end_local = _job.mid_joint.unwrap().invert(None) * *_job.end_joint.unwrap();

        // Rebuild corrected model transforms
        let start_correction = Float4x4::from_quaternion(_job.start_joint_correction.as_ref().unwrap().xyzw);
        let start_corrected = *_job.start_joint.unwrap() * start_correction;
        let mid_correction = Float4x4::from_quaternion(_job.mid_joint_correction.as_ref().unwrap().xyzw);
        let mid_corrected = start_corrected * mid_local * mid_correction;
        let end_corrected = mid_corrected * end_local;

        let diff = (end_corrected.cols[3] - _job.target).length3();
        assert_eq!(diff.get_x() < 1e-2, _reachable);

        assert_eq!(_job.reached.is_none() || **_job.reached.as_ref().unwrap() == _reachable, true);
    }

    #[test]
    fn job_validity() {
        todo!()
    }

    #[test]
    fn start_joint_correction() {
        todo!()
    }

    #[test]
    fn pole() {
        todo!()
    }

    #[test]
    fn zero_scale() {
        todo!()
    }

    #[test]
    fn soften() {
        todo!()
    }

    #[test]
    fn twist() {
        todo!()
    }

    #[test]
    fn weight() {
        todo!()
    }

    #[test]
    fn pole_target_alignment() {
        todo!()
    }

    #[test]
    fn mid_axis() {
        todo!()
    }

    #[test]
    fn aligned_joints_and_target() {
        todo!()
    }

    #[test]
    fn zero_length_start_target() {
        todo!()
    }

    #[test]
    fn zero_length_bone_chain() {
        todo!()
    }
}