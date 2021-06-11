/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::raw_skeleton::{RawSkeleton, SkeletonVisitor};
use crate::skeleton::Skeleton;

// Defines the class responsible of building Skeleton instances.
pub struct SkeletonBuilder {}

impl SkeletonBuilder {
    // Creates a Skeleton based on _raw_skeleton and *this builder parameters.
    // Returns a Skeleton instance on success, an empty unique_ptr on failure. See
    // RawSkeleton::Validate() for more details about failure reasons.
    // The skeleton is returned as an unique_ptr as ownership is given back to the
    // caller.
    pub fn apply(_raw_skeleton: &RawSkeleton) -> Skeleton {
        todo!()
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
    pub fn new(_num_joints: i32) -> JointLister<'a> {
        let mut result = JointLister {
            linear_joints: vec![]
        };
        result.linear_joints.reserve(_num_joints as usize);
        return result;
    }
}

impl<'a> SkeletonVisitor for JointLister<'a> {
    fn visitor(&mut self, _current: &crate::raw_skeleton::Joint,
               _parent: Option<&crate::raw_skeleton::Joint>) {
        // Looks for the "lister" parent.
        let mut parent = crate::skeleton::Constants::KNoParent as i16;
        if _parent.is_some() {
            // Start searching from the last joint.
            let mut j = self.linear_joints.len() as i16 - 1;
            while j >= 0 {
                if self.linear_joints[j as usize].joint == _parent.unwrap() {
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