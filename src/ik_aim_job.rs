/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::simd_math::*;
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
        if !self.validate() {
            return false;
        }

        // If matrices aren't invertible, they'll be all 0 (ozz::math
        // implementation), which will result in identity correction quaternions.
        let mut invertible = SimdInt4::zero();
        let inv_joint = self.joint.unwrap().invert(Some(&mut invertible));

        // Computes joint to target vector, in joint local-space (_js).
        let joint_to_target_js = inv_joint.transform_point(self.target);
        let joint_to_target_js_len2 = joint_to_target_js.length3sqr();

        // Recomputes forward vector to account for offset.
        // If the offset is further than target, it won't be reachable.
        let mut offsetted_forward = SimdFloat4::zero();
        let lreached = compute_offsetted_forward(self.forward, self.offset, joint_to_target_js,
                                                 &mut offsetted_forward);
        // Copies reachability result.
        // If offsetted forward vector doesn't exists, target position cannot be
        // aimed.
        if self.reached.is_some() {
            **self.reached.as_mut().unwrap() = lreached;
        }

        if !lreached || joint_to_target_js_len2.cmp_eq(SimdFloat4::zero()).are_all_true1() {
            // Target can't be reached or is too close to joint position to find a
            // direction.
            **self.joint_correction.as_mut().unwrap() = SimdQuaternion::identity();
            return true;
        }

        // Calculates joint_to_target_rot_ss quaternion which solves for
        // offsetted_forward vector rotating onto the target.
        let joint_to_target_rot_js = SimdQuaternion::from_vectors(offsetted_forward, joint_to_target_js);

        // Calculates rotate_plane_js quaternion which aligns joint up to the pole
        // vector.
        let corrected_up_js = joint_to_target_rot_js.transform_vector(self.up);

        // Compute (and normalize) reference and pole planes normals.
        let pole_vector_js = inv_joint.transform_vector(self.pole_vector);
        let ref_joint_normal_js = pole_vector_js.cross3(joint_to_target_js);
        let joint_normal_js = corrected_up_js.cross3(joint_to_target_js);
        let ref_joint_normal_js_len2 = ref_joint_normal_js.length3sqr();
        let joint_normal_js_len2 = joint_normal_js.length3sqr();

        let denoms = joint_to_target_js_len2.set_y(joint_normal_js_len2).set_z(ref_joint_normal_js_len2);

        let rotate_plane_axis_js;
        let rotate_plane_js;
        // Computing rotation axis and plane requires valid normals.
        if denoms.cmp_ne(SimdFloat4::zero()).are_all_true3() {
            let rsqrts = joint_to_target_js_len2.set_y(joint_normal_js_len2).set_z(ref_joint_normal_js_len2).rsqrt_est_nr();

            // Computes rotation axis, which is either joint_to_target_js or
            // -joint_to_target_js depending on rotation direction.
            rotate_plane_axis_js = joint_to_target_js * rsqrts.splat_x();

            // Computes angle cosine between the 2 normalized plane normals.
            let rotate_plane_cos_angle = joint_normal_js * rsqrts.splat_y().dot3(ref_joint_normal_js * rsqrts.splat_z());
            let axis_flip = ref_joint_normal_js.dot3(corrected_up_js).splat_x().and_fi(SimdInt4::mask_sign());
            let rotate_plane_axis_flipped_js = rotate_plane_axis_js.xor_ff(axis_flip);

            // Builds quaternion along rotation axis.
            let one = SimdFloat4::one();
            rotate_plane_js = SimdQuaternion::from_axis_cos_angle(
                rotate_plane_axis_flipped_js, rotate_plane_cos_angle.clamp(-one, one));
        } else {
            rotate_plane_axis_js = joint_to_target_js * denoms.rsqrt_est_xnr().splat_x();
            rotate_plane_js = SimdQuaternion::identity();
        }

        // Twists rotation plane.
        let twisted;
        if self.twist_angle != 0.0 {
            // If a twist angle is provided, rotation angle is rotated around joint to
            // target vector.
            let twist_ss = SimdQuaternion::from_axis_angle(
                rotate_plane_axis_js, SimdFloat4::load1(self.twist_angle));
            twisted = twist_ss * rotate_plane_js * joint_to_target_rot_js;
        } else {
            twisted = rotate_plane_js * joint_to_target_rot_js;
        }

        // Weights output quaternion.

        // Fix up quaternions so w is always positive, which is required for NLerp
        // (with identity quaternion) to lerp the shortest path.
        let twisted_fu = twisted.xyzw.xor_fi(SimdInt4::mask_sign().and(twisted.xyzw.splat_w().cmp_lt(SimdFloat4::zero())));

        if self.weight < 1.0 {
            // NLerp start and mid joint rotations.
            let identity = SimdFloat4::w_axis();
            let simd_weight = SimdFloat4::load1(self.weight).max0();
            (**self.joint_correction.as_mut().unwrap()).xyzw = identity.lerp(twisted.xyzw, simd_weight).normalize_est4();
        } else {
            // Quaternion doesn't need interpolation
            (**self.joint_correction.as_mut().unwrap()).xyzw = twisted_fu;
        }

        return true;
    }
}

// When there's an offset, the forward vector needs to be recomputed.
// The idea is to find the vector that will allow the point at offset position
// to aim at target position. This vector starts at joint position. It ends on a
// line perpendicular to pivot-offset line, at the intersection with the sphere
// defined by target position (centered on joint position). See geogebra
// diagram: media/doc/src/ik_aim_offset.ggb
fn compute_offsetted_forward(_forward: SimdFloat4, _offset: SimdFloat4,
                             _target: SimdFloat4, _offsetted_forward: &mut SimdFloat4) -> bool {
    // AO is projected offset vector onto the normalized forward vector.
    debug_assert!(_forward.is_normalized_est3().are_all_true1());
    let aol = _forward.dot3(_offset);

    // Compute square length of ac using Pythagorean theorem.
    let acl2 = _offset.length3sqr() - aol * aol;

    // Square length of target vector, aka circle radius.
    let r2 = _target.length3sqr();

    // If offset is outside of the sphere defined by target length, the target
    // isn't reachable.
    if acl2.cmp_gt(r2).are_all_true1() {
        return false;
    }

    // ail is the length of the vector from offset to sphere intersection.
    let ail = (r2 - acl2).sqrt_x();

    // The distance from offset position to the intersection with the sphere is
    // (ail - aol) Intersection point on the sphere can thus be computed.
    *_offsetted_forward = _offset + _forward * (ail - aol).splat_x();

    return true;
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod ik_aim_job {
    use crate::simd_quaternion::SimdQuaternion;
    use crate::ik_aim_job::IKAimJob;
    use crate::math_constant::*;
    use crate::quaternion::Quaternion;
    use crate::vec_float::Float3;
    use crate::math_test_helper::*;
    use crate::simd_math::*;
    use crate::*;

    #[test]
    fn job_validity() {
        let joint = Float4x4::identity();
        let mut quat = SimdQuaternion::identity();

        {  // Default is invalid
            let job = IKAimJob::new();
            assert_eq!(job.validate(), false);
        }

        {  // Invalid joint matrix
            let mut job = IKAimJob::new();
            job.joint = Some(&joint);
            assert_eq!(job.validate(), false);
        }

        {  // Invalid output
            let mut job = IKAimJob::new();
            job.joint_correction = Some(&mut quat);
            assert_eq!(job.validate(), false);
        }

        {  // Invalid non normalized forward vector.
            let mut job = IKAimJob::new();
            job.forward = SimdFloat4::load(0.5, 0.0, 0.0, 0.0);
            assert_eq!(job.validate(), false);
        }

        {  // Valid
            let mut job = IKAimJob::new();
            job.joint = Some(&joint);
            job.joint_correction = Some(&mut quat);
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }
    }

    #[test]
    fn correction() {
        let mut quat = SimdQuaternion::identity();

        let mut job = IKAimJob::new();
        job.joint_correction = Some(&mut quat);

        // Test will be executed with different root transformations
        let parents = [
            Float4x4::identity(),  // No root transformation
            Float4x4::translation(SimdFloat4::y_axis()),  // Up
            Float4x4::from_euler(SimdFloat4::load(K_PI / 3.0, 0.0, 0.0, 0.0)),  // Rotated
            Float4x4::scaling(SimdFloat4::load(2.0, 2.0, 2.0, 0.0)),  // Uniformly scaled
            Float4x4::scaling(SimdFloat4::load(1.0, 2.0, 1.0, 0.0)),  // Non-uniformly scaled
            Float4x4::scaling(SimdFloat4::load(-3.0, -3.0, -3.0, 0.0))  // Mirrored
        ];

        for i in 0..parents.len() {
            let parent = &parents[i];
            job.joint = Some(&parent);

            // These are in joint local-space
            job.forward = SimdFloat4::x_axis();
            job.up = SimdFloat4::y_axis();

            // Pole vector is in model space
            job.pole_vector = parent.transform_vector(SimdFloat4::y_axis());

            {  // x
                job.target = parent.transform_point(SimdFloat4::x_axis());
                assert_eq!(job.run(), true);
                expect_simd_quaternion_eq_tol!(quat, 0.0, 0.0, 0.0, 1.0, 2e-3);
            }

            {  // -x
                job.target = parent.transform_point(-SimdFloat4::x_axis());
                assert_eq!(job.run(), true);
                let y_pi = Quaternion::from_axis_angle(
                    &Float3::y_axis(), K_PI);
                expect_simd_quaternion_eq_tol!(quat, y_pi.x, y_pi.y, y_pi.z, y_pi.w, 2e-3);
            }

            {  // z
                job.target = parent.transform_point(SimdFloat4::z_axis());
                assert_eq!(job.run(), true);
                let y_m_pi_2 = Quaternion::from_axis_angle(&Float3::y_axis(), -K_PI_2);
                expect_simd_quaternion_eq_tol!(quat, y_m_pi_2.x, y_m_pi_2.y, y_m_pi_2.z,
                                             y_m_pi_2.w, 2e-3);
            }

            {  // -z
                job.target = parent.transform_point(-SimdFloat4::z_axis());
                assert_eq!(job.run(), true);
                let y_pi_2 = Quaternion::from_axis_angle(&Float3::y_axis(), K_PI_2);
                expect_simd_quaternion_eq_tol!(quat, y_pi_2.x, y_pi_2.y, y_pi_2.z, y_pi_2.w,
                                             2e-3);
            }

            {  // 45 up y
                job.target = parent.transform_point(SimdFloat4::load(1.0, 1.0, 0.0, 0.0));
                assert_eq!(job.run(), true);
                let z_pi_4 = Quaternion::from_axis_angle(&Float3::z_axis(), K_PI_4);
                expect_simd_quaternion_eq_tol!(quat, z_pi_4.x, z_pi_4.y, z_pi_4.z, z_pi_4.w,
                                             2e-3);
            }

            {  // 45 up y, further
                job.target = parent.transform_point(SimdFloat4::load(2.0, 2.0, 0.0, 0.0));
                assert_eq!(job.run(), true);
                let z_pi_4 = Quaternion::from_axis_angle(&Float3::z_axis(), K_PI_4);
                expect_simd_quaternion_eq_tol!(quat, z_pi_4.x, z_pi_4.y, z_pi_4.z, z_pi_4.w,
                                             2e-3);
            }
        }
    }

    #[test]
    fn forward() {
        let mut job = IKAimJob::new();
        let mut quat = SimdQuaternion::identity();
        job.joint_correction = Some(&mut quat);
        let joint = Float4x4::identity();
        job.joint = Some(&joint);

        job.target = SimdFloat4::x_axis();
        job.up = SimdFloat4::y_axis();
        job.pole_vector = SimdFloat4::y_axis();

        {  // forward x
            job.forward = SimdFloat4::x_axis();
            assert_eq!(job.run(), true);
            expect_simd_quaternion_eq_tol!(quat, 0.0, 0.0, 0.0, 1.0, 2e-3);
        }

        {  // forward -x
            job.forward = -SimdFloat4::x_axis();
            assert_eq!(job.run(), true);
            let y_pi = Quaternion::from_axis_angle(&Float3::y_axis(), -K_PI);
            expect_simd_quaternion_eq_tol!(quat, y_pi.x, y_pi.y, y_pi.z, y_pi.w, 2e-3);
        }

        {  // forward z
            job.forward = SimdFloat4::z_axis();
            assert_eq!(job.run(), true);
            let y_pi_2 = Quaternion::from_axis_angle(&Float3::y_axis(), K_PI_2);
            expect_simd_quaternion_eq_tol!(quat, y_pi_2.x, y_pi_2.y, y_pi_2.z, y_pi_2.w,
                                         2e-3);
        }
    }

    #[test]
    fn up() {
        let mut job = IKAimJob::new();
        let mut quat = SimdQuaternion::identity();
        job.joint_correction = Some(&mut quat);
        let joint = Float4x4::identity();
        job.joint = Some(&joint);

        job.target = SimdFloat4::x_axis();
        job.forward = SimdFloat4::x_axis();
        job.up = SimdFloat4::y_axis();
        job.pole_vector = SimdFloat4::y_axis();

        {  // up y
            job.up = SimdFloat4::y_axis();
            assert_eq!(job.run(), true);
            expect_simd_quaternion_eq_tol!(quat, 0.0, 0.0, 0.0, 1.0, 2e-3);
        }

        {  // up -y
            job.up = -SimdFloat4::y_axis();
            assert_eq!(job.run(), true);
            let x_pi = Quaternion::from_axis_angle(&Float3::x_axis(), K_PI);
            expect_simd_quaternion_eq_tol!(quat, x_pi.x, x_pi.y, x_pi.z, x_pi.w, 2e-3);
        }

        {  // up z
            job.up = SimdFloat4::z_axis();
            assert_eq!(job.run(), true);
            let x_m_pi_2 = Quaternion::from_axis_angle(&Float3::x_axis(), -K_PI_2);
            expect_simd_quaternion_eq_tol!(quat, x_m_pi_2.x, x_m_pi_2.y, x_m_pi_2.z,
                                         x_m_pi_2.w, 2e-3);
        }

        {  // up 2*z
            job.up = SimdFloat4::z_axis() * SimdFloat4::load1(2.0);
            assert_eq!(job.run(), true);
            let x_m_pi_2 = Quaternion::from_axis_angle(&Float3::x_axis(), -K_PI_2);
            expect_simd_quaternion_eq_tol!(quat, x_m_pi_2.x, x_m_pi_2.y, x_m_pi_2.z,
                                         x_m_pi_2.w, 2e-3);
        }

        {  // up very small z
            job.up = SimdFloat4::z_axis() * SimdFloat4::load1(1e-9);
            assert_eq!(job.run(), true);
            let x_m_pi_2 = Quaternion::from_axis_angle(&Float3::x_axis(), -K_PI_2);
            expect_simd_quaternion_eq_tol!(quat, x_m_pi_2.x, x_m_pi_2.y, x_m_pi_2.z,
                                         x_m_pi_2.w, 2e-3);
        }

        {  // up is zero
            job.up = SimdFloat4::zero();
            assert_eq!(job.run(), true);
            expect_simd_quaternion_eq_tol!(quat, 0.0, 0.0, 0.0, 1.0, 2e-3);
        }
    }

    #[test]
    fn pole() {
        let mut job = IKAimJob::new();
        let mut quat = SimdQuaternion::identity();
        job.joint_correction = Some(&mut quat);
        let joint = Float4x4::identity();
        job.joint = Some(&joint);

        job.target = SimdFloat4::x_axis();
        job.forward = SimdFloat4::x_axis();
        job.up = SimdFloat4::y_axis();

        {  // Pole y
            job.pole_vector = SimdFloat4::y_axis();
            assert_eq!(job.run(), true);
            expect_simd_quaternion_eq_tol!(quat, 0.0, 0.0, 0.0, 1.0, 2e-3);
        }

        {  // Pole -y
            job.pole_vector = -SimdFloat4::y_axis();
            assert_eq!(job.run(), true);
            let x_pi = Quaternion::from_axis_angle(&Float3::x_axis(), K_PI);
            expect_simd_quaternion_eq_tol!(quat, x_pi.x, x_pi.y, x_pi.z, x_pi.w, 2e-3);
        }

        {  // Pole z
            job.pole_vector = SimdFloat4::z_axis();
            assert_eq!(job.run(), true);
            let x_pi_2 = Quaternion::from_axis_angle(&Float3::x_axis(), K_PI_2);
            expect_simd_quaternion_eq_tol!(quat, x_pi_2.x, x_pi_2.y, x_pi_2.z, x_pi_2.w,
                                         2e-3);
        }

        {  // Pole 2*z
            job.pole_vector = SimdFloat4::z_axis() * SimdFloat4::load1(2.0);
            assert_eq!(job.run(), true);
            let x_pi_2 = Quaternion::from_axis_angle(&Float3::x_axis(), K_PI_2);
            expect_simd_quaternion_eq_tol!(quat, x_pi_2.x, x_pi_2.y, x_pi_2.z, x_pi_2.w,
                                         2e-3);
        }

        {  // Pole very small z
            job.pole_vector = SimdFloat4::z_axis() * SimdFloat4::load1(1e-9);
            assert_eq!(job.run(), true);
            let x_pi_2 = Quaternion::from_axis_angle(&Float3::x_axis(), K_PI_2);
            expect_simd_quaternion_eq_tol!(quat, x_pi_2.x, x_pi_2.y, x_pi_2.z, x_pi_2.w,
                                         2e-3);
        }
    }

    #[test]
    fn offset() {
        let mut job = IKAimJob::new();
        let mut quat = SimdQuaternion::identity();
        job.joint_correction = Some(&mut quat);
        let joint = Float4x4::identity();
        job.joint = Some(&joint);
        let mut reached = true;
        job.reached = Some(&mut reached);

        job.target = SimdFloat4::x_axis();
        job.forward = SimdFloat4::x_axis();
        job.up = SimdFloat4::y_axis();
        job.pole_vector = SimdFloat4::y_axis();

        {  // No offset
            reached = false;
            job.offset = SimdFloat4::zero();
            assert_eq!(job.run(), true);
            expect_simd_quaternion_eq_tol!(quat, 0.0, 0.0, 0.0, 1.0, 2e-3);
            assert_eq!(reached, true);
        }

        {  // Offset inside target sphere
            reached = false;
            job.offset = SimdFloat4::load(0.0, K_SQRT2_2, 0.0, 0.0);
            assert_eq!(job.run(), true);
            let z_pi_4 = Quaternion::from_axis_angle(&Float3::z_axis(), -K_PI_4);
            expect_simd_quaternion_eq_tol!(quat, z_pi_4.x, z_pi_4.y, z_pi_4.z, z_pi_4.w,
                                         2e-3);
            assert_eq!(reached, true);
        }

        {  // Offset inside target sphere
            reached = false;
            job.offset = SimdFloat4::load(0.5, 0.5, 0.0, 0.0);
            assert_eq!(job.run(), true);
            let z_pi_6 = Quaternion::from_axis_angle(&Float3::z_axis(), -K_PI / 6.0);
            expect_simd_quaternion_eq_tol!(quat, z_pi_6.x, z_pi_6.y, z_pi_6.z, z_pi_6.w,
                                         2e-3);
            assert_eq!(reached, true);
        }

        {  // Offset inside target sphere
            reached = false;
            job.offset = SimdFloat4::load(-0.5, 0.5, 0.0, 0.0);
            assert_eq!(job.run(), true);
            let z_pi_6 = Quaternion::from_axis_angle(&Float3::z_axis(), -K_PI / 6.0);
            expect_simd_quaternion_eq_tol!(quat, z_pi_6.x, z_pi_6.y, z_pi_6.z, z_pi_6.w,
                                         2e-3);
            assert_eq!(reached, true);
        }

        {  // Offset inside target sphere
            reached = false;
            job.offset = SimdFloat4::load(0.5, 0.0, 0.5, 0.0);
            assert_eq!(job.run(), true);
            let y_pi_6 = Quaternion::from_axis_angle(&Float3::y_axis(), K_PI / 6.0);
            expect_simd_quaternion_eq_tol!(quat, y_pi_6.x, y_pi_6.y, y_pi_6.z, y_pi_6.w,
                                         2e-3);
            assert_eq!(reached, true);
        }

        {  // Offset on target sphere
            reached = false;
            job.offset = SimdFloat4::load(0.0, 1.0, 0.0, 0.0);
            assert_eq!(job.run(), true);
            let z_pi_2 = Quaternion::from_axis_angle(&Float3::z_axis(), -K_PI_2);
            expect_simd_quaternion_eq_tol!(quat, z_pi_2.x, z_pi_2.y, z_pi_2.z, z_pi_2.w,
                                         2e-3);
            assert_eq!(reached, true);
        }

        {  // Offset outside of target sphere, unreachable
            reached = true;
            job.offset = SimdFloat4::load(0.0, 2.0, 0.0, 0.0);
            assert_eq!(job.run(), true);
            expect_simd_quaternion_eq_tol!(quat, 0.0, 0.0, 0.0, 1.0, 2e-3);
            assert_eq!(reached, false);
        }

        let translated_joint = Float4x4::translation(SimdFloat4::y_axis());
        job.joint = Some(&translated_joint);

        {  // Offset inside of target sphere, unreachable
            reached = false;
            job.offset = SimdFloat4::y_axis();
            assert_eq!(job.run(), true);
            let z_pi_2 = Quaternion::from_axis_angle(&Float3::z_axis(), -K_PI_2);
            expect_simd_quaternion_eq_tol!(quat, z_pi_2.x, z_pi_2.y, z_pi_2.z, z_pi_2.w,
                                         2e-3);
            assert_eq!(reached, true);
        }
    }

    #[test]
    fn twist() {
        let mut job = IKAimJob::new();
        let mut quat = SimdQuaternion::identity();
        job.joint_correction = Some(&mut quat);
        let joint = Float4x4::identity();
        job.joint = Some(&joint);

        job.target = SimdFloat4::x_axis();
        job.forward = SimdFloat4::x_axis();
        job.up = SimdFloat4::y_axis();

        {  // Pole y, twist 0
            job.pole_vector = SimdFloat4::y_axis();
            job.twist_angle = 0.0;
            assert_eq!(job.run(), true);
            expect_simd_quaternion_eq_tol!(quat, 0.0, 0.0, 0.0, 1.0, 2e-3);
        }

        {  // Pole y, twist pi
            job.pole_vector = SimdFloat4::y_axis();
            job.twist_angle = K_PI;
            assert_eq!(job.run(), true);
            let x_pi = Quaternion::from_axis_angle(&Float3::x_axis(), -K_PI);
            expect_simd_quaternion_eq_tol!(quat, x_pi.x, x_pi.y, x_pi.z, x_pi.w, 2e-3);
        }

        {  // Pole y, twist -pi
            job.pole_vector = SimdFloat4::y_axis();
            job.twist_angle = -K_PI;
            assert_eq!(job.run(), true);
            let x_m_pi = Quaternion::from_axis_angle(&Float3::x_axis(), -K_PI);
            expect_simd_quaternion_eq_tol!(quat, x_m_pi.x, x_m_pi.y, x_m_pi.z, x_m_pi.w, 2e-3);
        }

        {  // Pole y, twist pi/2
            job.pole_vector = SimdFloat4::y_axis();
            job.twist_angle = K_PI_2;
            assert_eq!(job.run(), true);
            let x_pi_2 = Quaternion::from_axis_angle(&Float3::x_axis(), K_PI_2);
            expect_simd_quaternion_eq_tol!(quat, x_pi_2.x, x_pi_2.y, x_pi_2.z, x_pi_2.w, 2e-3);
        }

        {  // Pole z, twist pi/2
            job.pole_vector = SimdFloat4::z_axis();
            job.twist_angle = K_PI_2;
            assert_eq!(job.run(), true);
            let x_pi = Quaternion::from_axis_angle(&Float3::x_axis(), K_PI);
            expect_simd_quaternion_eq_tol!(quat, x_pi.x, x_pi.y, x_pi.z, x_pi.w, 2e-3);
        }
    }

    #[test]
    fn aligned_target_up() {
        let mut job = IKAimJob::new();
        let mut quat = SimdQuaternion::identity();
        job.joint_correction = Some(&mut quat);
        let joint = Float4x4::identity();
        job.joint = Some(&joint);

        job.forward = SimdFloat4::x_axis();
        job.pole_vector = SimdFloat4::y_axis();

        {  // Not aligned
            job.target = SimdFloat4::x_axis();
            job.up = SimdFloat4::y_axis();
            assert_eq!(job.run(), true);
            expect_simd_quaternion_eq_tol!(quat, 0.0, 0.0, 0.0, 1.0, 2e-3);
        }

        {  // Aligned y
            job.target = SimdFloat4::y_axis();
            job.up = SimdFloat4::y_axis();
            assert_eq!(job.run(), true);
            let z_pi_2 = Quaternion::from_axis_angle(&Float3::z_axis(), K_PI_2);
            expect_simd_quaternion_eq_tol!(quat, z_pi_2.x, z_pi_2.y, z_pi_2.z, z_pi_2.w,
                                         2e-3);
        }

        {  // Aligned 2*y
            job.target = SimdFloat4::y_axis() * SimdFloat4::load1(2.0);
            job.up = SimdFloat4::y_axis();
            assert_eq!(job.run(), true);
            let z_pi_2 = Quaternion::from_axis_angle(&Float3::z_axis(), K_PI_2);
            expect_simd_quaternion_eq_tol!(quat, z_pi_2.x, z_pi_2.y, z_pi_2.z, z_pi_2.w,
                                         2e-3);
        }

        {  // Aligned -2*y
            job.target = SimdFloat4::y_axis() * SimdFloat4::load1(-2.0);
            job.up = SimdFloat4::y_axis();
            assert_eq!(job.run(), true);
            let z_m_pi_2 = Quaternion::from_axis_angle(&Float3::z_axis(), -K_PI_2);
            expect_simd_quaternion_eq_tol!(quat, z_m_pi_2.x, z_m_pi_2.y, z_m_pi_2.z,
                                         z_m_pi_2.w, 2e-3);
        }
    }

    #[test]
    fn aligned_target_pole() {
        let mut job = IKAimJob::new();
        let mut quat = SimdQuaternion::identity();
        job.joint_correction = Some(&mut quat);
        let joint = Float4x4::identity();
        job.joint = Some(&joint);

        job.forward = SimdFloat4::x_axis();
        job.up = SimdFloat4::y_axis();

        {  // Not aligned
            job.target = SimdFloat4::x_axis();
            job.pole_vector = SimdFloat4::y_axis();
            assert_eq!(job.run(), true);
            expect_simd_quaternion_eq_tol!(quat, 0.0, 0.0, 0.0, 1.0, 2e-3);
        }

        {  // Aligned y
            job.target = SimdFloat4::y_axis();
            job.pole_vector = SimdFloat4::y_axis();
            assert_eq!(job.run(), true);
            let z_pi_2 = Quaternion::from_axis_angle(&Float3::z_axis(), K_PI_2);
            expect_simd_quaternion_eq_tol!(quat, z_pi_2.x, z_pi_2.y, z_pi_2.z, z_pi_2.w, 2e-3);
        }
    }

    #[test]
    fn target_too_close() {
        let mut job = IKAimJob::new();
        let mut quat = SimdQuaternion::identity();
        job.joint_correction = Some(&mut quat);
        let joint = Float4x4::identity();
        job.joint = Some(&joint);

        job.target = SimdFloat4::zero();
        job.forward = SimdFloat4::x_axis();
        job.up = SimdFloat4::y_axis();
        job.pole_vector = SimdFloat4::y_axis();

        assert_eq!(job.run(), true);
        expect_simd_quaternion_eq_tol!(quat, 0.0, 0.0, 0.0, 1.0, 2e-3);
    }

    #[test]
    fn weight() {
        let mut job = IKAimJob::new();
        let mut quat = SimdQuaternion::identity();
        job.joint_correction = Some(&mut quat);
        let joint = Float4x4::identity();
        job.joint = Some(&joint);

        job.target = SimdFloat4::z_axis();
        job.forward = SimdFloat4::x_axis();
        job.up = SimdFloat4::y_axis();
        job.pole_vector = SimdFloat4::y_axis();

        {  // Full weight
            job.weight = 1.0;
            assert_eq!(job.run(), true);
            let y_m_pi2 = Quaternion::from_axis_angle(&Float3::y_axis(), -K_PI_2);
            expect_simd_quaternion_eq_tol!(quat, y_m_pi2.x, y_m_pi2.y, y_m_pi2.z, y_m_pi2.w,
                                         2e-3);
        }

        {  // > 1
            job.weight = 2.0;
            assert_eq!(job.run(), true);
            let y_m_pi2 = Quaternion::from_axis_angle(&Float3::y_axis(), -K_PI_2);
            expect_simd_quaternion_eq_tol!(quat, y_m_pi2.x, y_m_pi2.y, y_m_pi2.z, y_m_pi2.w,
                                         2e-3);
        }

        {  // Half weight
            job.weight = 0.5;
            assert_eq!(job.run(), true);
            let y_m_pi4 = Quaternion::from_axis_angle(&Float3::y_axis(), -K_PI_4);
            expect_simd_quaternion_eq_tol!(quat, y_m_pi4.x, y_m_pi4.y, y_m_pi4.z, y_m_pi4.w,
                                         2e-3);
        }

        {  // Zero weight
            job.weight = 0.0;
            assert_eq!(job.run(), true);
            expect_simd_quaternion_eq_tol!(quat, 0.0, 0.0, 0.0, 1.0, 2e-3);
        }

        {  // < 0
            job.weight = -0.5;
            assert_eq!(job.run(), true);
            expect_simd_quaternion_eq_tol!(quat, 0.0, 0.0, 0.0, 1.0, 2e-3);
        }
    }

    #[test]
    fn zero_scale() {
        let mut job = IKAimJob::new();
        let mut quat = SimdQuaternion::identity();
        job.joint_correction = Some(&mut quat);
        let joint = Float4x4::scaling(SimdFloat4::zero());
        job.joint = Some(&joint);

        assert_eq!(job.run(), true);
        expect_simd_quaternion_eq_tol!(quat, 0.0, 0.0, 0.0, 1.0, 2e-3);
    }
}