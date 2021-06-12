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
    fn visitor(&self, _current: i32, _parent: i32);
}

// Applies a specified functor to each joint in a depth-first order.
// _Fct is of type void(int _current, int _parent) where the first argument is
// the child of the second argument. _parent is kNoParent if the
// _current joint is a root. _from indicates the joint from which the joint
// hierarchy traversal begins. Use Skeleton::kNoParent to traverse the
// whole hierarchy, in case there are multiple roots.
pub fn iterate_joints_df<_Fct: JointVisitor>(_skeleton: &Skeleton, _fct: _Fct, _from: Option<i32>) -> _Fct {
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
pub fn iterate_joints_df_reverse<_Fct: JointVisitor>(_skeleton: &Skeleton, _fct: _Fct) -> _Fct {
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

    #[test]
    fn joint_bind_pose() {
        todo!()
    }
}