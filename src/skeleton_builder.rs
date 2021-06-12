/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::raw_skeleton::*;
use crate::skeleton::Skeleton;
use crate::simd_math::SimdFloat4;

// Defines the class responsible of building Skeleton instances.
pub struct SkeletonBuilder {}

impl SkeletonBuilder {
    // Creates a Skeleton based on _raw_skeleton and *this builder parameters.
    // Returns a Skeleton instance on success, an empty unique_ptr on failure. See
    // RawSkeleton::Validate() for more details about failure reasons.
    // The skeleton is returned as an unique_ptr as ownership is given back to the
    // caller.
    pub fn apply<'a>(_raw_skeleton: &RawSkeleton, skeleton: &'a mut Skeleton<'a>) {
        // Everything is fine, allocates and fills the skeleton.
        // Will not fail.
        let num_joints = _raw_skeleton.num_joints();

        // Iterates through all the joint of the raw skeleton and fills a sorted joint
        // list.
        // Iteration order defines runtime skeleton joint ordering.
        let mut lister = JointLister::new(num_joints);
        let linear_joints = &iterate_joints_df(_raw_skeleton, &mut lister).linear_joints;
        debug_assert!(linear_joints.len() as i32 == num_joints);

        // Computes name's buffer size.
        for i in 0..num_joints {
            let current = linear_joints[i as usize].joint;
            skeleton.whole_name += &*current.name;
        }

        // Allocates all skeleton members.
        skeleton.allocate(num_joints as usize);

        // Copy names. All names are allocated in a single buffer. Only the first name
        // is set, all other names array entries must be initialized.
        let mut start = 0;
        let mut end = 0;
        for i in 0..num_joints {
            let current = linear_joints[i as usize].joint;
            end += current.name.len();
            skeleton.joint_names_[i as usize] = &(skeleton.whole_name[start..end]);
            start = end;
        }

        // Transfers sorted joints hierarchy to the new skeleton.
        for i in 0..num_joints {
            skeleton.joint_parents_[i as usize] = linear_joints[i as usize].parent;
        }

        // Transfers t-poses.
        let w_axis = SimdFloat4::w_axis();
        let zero = SimdFloat4::zero();
        let one = SimdFloat4::one();

        for i in 0..skeleton.num_soa_joints() {
            let mut translations = [SimdFloat4::zero(); 4];
            let mut scales = [SimdFloat4::zero(); 4];
            let mut rotations = [SimdFloat4::zero(); 4];
            for j in 0..4 {
                if i * 4 + j < num_joints {
                    let src_joint = linear_joints[(i * 4 + j) as usize].joint;
                    translations[j as usize] = SimdFloat4::load3ptr_u(src_joint.transform.translation.to_vec4());
                    rotations[j as usize] = SimdFloat4::load_ptr_u(src_joint.transform.rotation.to_vec()).normalize_safe4(w_axis);
                    scales[j as usize] = SimdFloat4::load3ptr_u(src_joint.transform.scale.to_vec4());
                } else {
                    translations[j as usize] = zero;
                    rotations[j as usize] = w_axis;
                    scales[j as usize] = one;
                }
            }
            // Fills the SoaTransform structure.
            SimdFloat4::transpose4x3(&translations,
                                     &mut skeleton.joint_bind_poses_[i as usize].translation);
            SimdFloat4::transpose4x4(&rotations, &mut skeleton.joint_bind_poses_[i as usize].rotation);
            SimdFloat4::transpose4x3(&scales, &mut skeleton.joint_bind_poses_[i as usize].scale);
        }

        return;  // Success.
    }
}

//--------------------------------------------------------------------------------------------------
struct Joint<'a> {
    joint: &'a crate::raw_skeleton::Joint,
    parent: i16,
}

// Stores each traversed joint in a vector.
struct JointLister<'a> {
    // Array of joints in the traversed DAG order.
    linear_joints: Vec<Joint<'a>>,
}

impl<'a> JointLister<'a> {
    fn new(_num_joints: i32) -> JointLister<'a> {
        let mut result = JointLister {
            linear_joints: vec![]
        };
        result.linear_joints.reserve(_num_joints as usize);
        return result;
    }
}

impl<'a> SkeletonVisitor<'a> for JointLister<'a> {
    fn visitor(&mut self, _current: &'a crate::raw_skeleton::Joint,
               _parent: Option<&crate::raw_skeleton::Joint>) {
        // Looks for the "lister" parent.
        let mut parent = crate::skeleton::Constants::KNoParent as i16;
        if _parent.is_some() {
            // Start searching from the last joint.
            let mut j = self.linear_joints.len() as i16 - 1;
            while j >= 0 {
                if self.linear_joints[j as usize].joint as *const _ == _parent.unwrap() as *const _ {
                    parent = j;
                    break;
                }
                j -= 1;
            }
            debug_assert!(parent >= 0);
        }
        let listed = Joint { joint: _current, parent };
        self.linear_joints.push(listed);
    }
}