/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::transform::Transform;
use crate::skeleton::Skeleton;
use crate::simd_math::SimdFloat4;
use crate::vec_float::Float3;
use crate::quaternion::Quaternion;

// Get bind-pose of a skeleton joint.
pub fn get_joint_local_bind_pose(_skeleton: &Skeleton, _joint: i32) -> Transform {
    debug_assert!(_joint >= 0 && _joint < _skeleton.num_joints() &&
        "Joint index out of range.".parse().unwrap_or(true));

    let soa_transform = _skeleton.joint_bind_poses()[_joint as usize / 4];

    // Transpose SoA data to AoS.
    let mut translations = [SimdFloat4::zero(); 4];
    SimdFloat4::transpose3x4(&soa_transform.translation, &mut translations);
    let mut rotations = [SimdFloat4::zero(); 4];
    SimdFloat4::transpose4x4_from_quat(&soa_transform.rotation, &mut rotations);
    let mut scales = [SimdFloat4::zero(); 4];
    SimdFloat4::transpose3x4(&soa_transform.scale, &mut scales);

    // Stores to the Transform object.
    let mut bind_pose = Transform::identity();
    let offset = _joint as usize % 4;

    let mut result = [0.0_f32; 4];
    SimdFloat4::store3ptr_u(&translations[offset], &mut result);
    bind_pose.translation = Float3::new(result[0], result[1], result[2]);
    SimdFloat4::store_ptr_u(&rotations[offset], &mut result);
    bind_pose.rotation = Quaternion::new(result[0], result[1], result[2], result[3]);
    SimdFloat4::store3ptr_u(&scales[offset], &mut result);
    bind_pose.scale = Float3::new(result[0], result[1], result[2]);

    return bind_pose;
}

// Test if a joint is a leaf. _joint number must be in range [0, num joints].
// "_joint" is a leaf if it's the last joint, or next joint's parent isn't
// "_joint".
pub fn is_leaf(_skeleton: &Skeleton, _joint: i32) -> bool {
    let num_joints = _skeleton.num_joints();
    debug_assert!(_joint >= 0 && _joint < num_joints && "_joint index out of range".parse().unwrap_or(true));
    let parents = _skeleton.joint_parents();
    let next = _joint + 1;
    return next == num_joints || parents[next as usize] != _joint as i16;
}

pub trait JointVisitor {
    fn visitor(&mut self, _current: i32, _parent: i32);
}

// Applies a specified functor to each joint in a depth-first order.
// _Fct is of type void(int _current, int _parent) where the first argument is
// the child of the second argument. _parent is kNoParent if the
// _current joint is a root. _from indicates the joint from which the joint
// hierarchy traversal begins. Use Skeleton::kNoParent to traverse the
// whole hierarchy, in case there are multiple roots.
pub fn iterate_joints_df<_Fct: JointVisitor>(_skeleton: &Skeleton, mut _fct: _Fct, _from: Option<i32>) -> _Fct {
    let parents = _skeleton.joint_parents();
    let num_joints = _skeleton.num_joints();

    let _from = _from.unwrap_or(crate::skeleton::Constants::KNoParent as i32);
    let mut i = match _from < 0 {
        true => 0,
        false => _from,
    };
    let mut process = i < num_joints;
    while process {
        _fct.visitor(i, parents[i as usize] as i32);
        i += 1;
        process = i < num_joints && parents[i as usize] >= _from as i16
    }

    return _fct;
}

// Applies a specified functor to each joint in a reverse (from leaves to root)
// depth-first order. _Fct is of type void(int _current, int _parent) where the
// first argument is the child of the second argument. _parent is kNoParent if
// the _current joint is a root.
pub fn iterate_joints_df_reverse<_Fct: JointVisitor>(_skeleton: &Skeleton, mut _fct: _Fct) -> _Fct {
    let parents = _skeleton.joint_parents();
    let mut i = _skeleton.num_joints() - 1;
    while i >= 0 {
        _fct.visitor(i, parents[i as usize] as i32);
        i -= 1;
    }

    return _fct;
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod skeleton_utils {
    use crate::skeleton_builder::SkeletonBuilder;
    use crate::raw_skeleton::*;
    use crate::transform::Transform;
    use crate::vec_float::Float3;
    use crate::quaternion::Quaternion;
    use crate::simd_math::SimdFloat4;
    use crate::math_test_helper::*;
    use crate::simd_math::*;
    use crate::*;
    use crate::skeleton_utils::*;

    #[test]
    fn joint_bind_pose() {
        let mut raw_skeleton = RawSkeleton::new();
        raw_skeleton.roots.resize(1, Joint::new());
        let r = &mut raw_skeleton.roots[0];
        r.name = "r0".to_string();
        r.transform.translation = Float3::x_axis();
        r.transform.rotation = Quaternion::identity();
        r.transform.scale = Float3::zero();

        r.children.resize(2, Joint::new());
        let c0 = &mut r.children[0];
        c0.name = "j0".to_string();
        c0.transform.translation = Float3::y_axis();
        c0.transform.rotation = -Quaternion::identity();
        c0.transform.scale = -Float3::one();

        let c1 = &mut r.children[1];
        c1.name = "j1".to_string();
        c1.transform.translation = Float3::z_axis();
        c1.transform.rotation = Quaternion::identity().conjugate();
        c1.transform.scale = Float3::one();

        assert_eq!(raw_skeleton.validate(), true);
        assert_eq!(raw_skeleton.num_joints(), 3);

        let skeleton = SkeletonBuilder::apply(&raw_skeleton);
        assert_eq!(skeleton.is_some(), true);
        assert_eq!(skeleton.as_ref().unwrap().num_joints(), 3);

        // Out of range.
        // EXPECT_ASSERTION(GetJointLocalBindPose(*skeleton, 3),
        //                  "Joint index out of range.");

        let bind_pose0 = get_joint_local_bind_pose(skeleton.as_ref().unwrap(), 0);
        expect_float3_eq!(bind_pose0.translation, 1.0, 0.0, 0.0);
        expect_quaternion_eq!(bind_pose0.rotation, 0.0, 0.0, 0.0, 1.0);
        expect_float3_eq!(bind_pose0.scale, 0.0, 0.0, 0.0);

        let bind_pose1 = get_joint_local_bind_pose(skeleton.as_ref().unwrap(), 1);
        expect_float3_eq!(bind_pose1.translation, 0.0, 1.0, 0.0);
        expect_quaternion_eq!(bind_pose1.rotation, 0.0, 0.0, 0.0, -1.0);
        expect_float3_eq!(bind_pose1.scale, -1.0, -1.0, -1.0);

        let bind_pose2 = get_joint_local_bind_pose(skeleton.as_ref().unwrap(), 2);
        expect_float3_eq!(bind_pose2.translation, 0.0, 0.0, 1.0);
        expect_quaternion_eq!(bind_pose2.rotation, -0.0, -0.0, -0.0, 1.0);
        expect_float3_eq!(bind_pose2.scale, 1.0, 1.0, 1.0);
    }

    /* Definition of the skeleton used by the tests.
     10 joints, 2 roots

          *
        /   \
       j0    j8
     /   \     \
     j1   j4    j9
     |   / \
     j2 j5 j6
     |     |
     j3    j7
     */
    struct IterateDFFailTester {}

    impl JointVisitor for IterateDFFailTester {
        fn visitor(&mut self, _current: i32, _parent: i32) {
            assert_eq!(false, true);
        }
    }

    struct IterateDFTester<'a> {
        // Iterated skeleton.
        skeleton_: &'a Skeleton,

        // First joint to explore.
        start_: i32,

        // Number of iterations completed.
        num_iterations_: i32,
    }

    impl<'a> IterateDFTester<'a> {
        pub fn new(_skeleton: &'a Skeleton, _start: i32) -> IterateDFTester<'a> {
            return IterateDFTester {
                skeleton_: _skeleton,
                start_: _start,
                num_iterations_: 0,
            };
        }

        pub fn num_iterations(&self) -> i32 { return self.num_iterations_; }
    }

    impl<'a> JointVisitor for IterateDFTester<'a> {
        fn visitor(&mut self, _current: i32, _parent: i32) {
            let joint = self.start_ + self.num_iterations_;
            assert_eq!(joint, _current);
            assert_eq!(self.skeleton_.joint_parents()[joint as usize], _parent as i16);
            self.num_iterations_ += 1;
        }
    }
}