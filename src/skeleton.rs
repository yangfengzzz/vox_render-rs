/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::skeleton::Constants::KMaxJoints;
use crate::soa_transform::SoaTransform;

// Defines Skeleton constant values.
pub enum Constants {
    // Defines the maximum number of joints.
    // This is limited in order to control the number of bits required to store
    // a joint index. Limiting the number of joints also helps handling worst
    // size cases, like when it is required to allocate an array of joints on
    // the stack.
    KMaxJoints = 1024,

    // Defines the maximum number of SoA elements required to store the maximum
    // number of joints.
    KMaxSoAJoints = (KMaxJoints as isize + 3) / 4,

    // Defines the index of the parent of the root joint (which has no parent in
    // fact).
    KNoParent = -1,
}

// This runtime skeleton data structure provides a const-only access to joint
// hierarchy, joint names and bind-pose. This structure is filled by the
// SkeletonBuilder and can be serialize/deserialized.
// Joint names, bind-poses and hierarchy information are all stored in separate
// arrays of data (as opposed to joint structures for the RawSkeleton), in order
// to closely match with the way runtime algorithms use them. Joint hierarchy is
// packed as an array of parent jont indices (16 bits), stored in depth-first
// order. This is enough to traverse the whole joint hierarchy. See
// iterate_joints_df() from skeleton_utils.h that implements a depth-first
// traversal utility.
pub struct Skeleton {
    // Bind pose of every joint in local space.
    pub joint_bind_poses_: Vec<SoaTransform>,

    // Array of joint parent indexes.
    pub joint_parents_: Vec<i16>,

    // Stores the name of every joint in an array of c-strings.
    pub joint_names_: Vec<String>,
}

impl Skeleton {
    // Builds a default skeleton.
    pub fn new() -> Skeleton {
        return Skeleton {
            joint_bind_poses_: vec![],
            joint_parents_: vec![],
            joint_names_: vec![],
        };
    }

    // Returns the number of joints of *this skeleton.
    pub fn num_joints(&self) -> i32 { return self.joint_parents_.len() as i32; }

    // Returns the number of soa elements matching the number of joints of *this
    // skeleton. This value is useful to allocate SoA runtime data structures.
    pub fn num_soa_joints(&self) -> i32 { return (self.num_joints() + 3) / 4; }

    // Returns joint's bind poses. Bind poses are stored in soa format.
    pub fn joint_bind_poses(&self) -> &Vec<SoaTransform> {
        return &self.joint_bind_poses_;
    }

    // Returns joint's parent indices range.
    pub fn joint_parents(&self) -> &Vec<i16> {
        return &self.joint_parents_;
    }

    // Returns joint's name collection.
    pub fn joint_names(&self) -> &Vec<String> {
        return &self.joint_names_;
    }
}

impl Skeleton {
    // Internal allocation/deallocation function.
    // allocate returns the beginning of the contiguous buffer of names.
    pub fn allocate(&mut self, _num_joints: usize) {
        debug_assert!(self.joint_bind_poses_.len() == 0 && self.joint_names_.len() == 0 &&
            self.joint_parents_.len() == 0);

        // Early out if no joint.
        if _num_joints == 0 {
            return;
        }

        // Bind poses have SoA format
        let num_soa_joints = (_num_joints + 3) / 4;

        // Serves larger alignment values first.
        // Bind pose first, biggest alignment.
        self.joint_bind_poses_.resize(num_soa_joints, SoaTransform::identity());

        // Then names array, second biggest alignment.
        self.joint_names_.resize(_num_joints, "".to_string());

        // Parents, third biggest alignment.
        self.joint_parents_.resize(_num_joints, 0);
    }
}