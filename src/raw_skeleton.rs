/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::transform::Transform;

// Offline skeleton joint type.
struct Joint {
    // Children joints.
    children: Vec<Joint>,

    // The name of the joint.
    name: String,

    // Joint bind pose transformation in local space.
    transform: Transform,
}

// Off-line skeleton type.
// This skeleton type is not intended to be used in run time. It is used to
// define the offline skeleton object that can be converted to the runtime
// skeleton using the SkeletonBuilder. This skeleton structure exposes joints'
// hierarchy. A joint is defined with a name, a transformation (its bind pose),
// and its children. Children are exposed as a public std::vector of joints.
// This same type is used for skeleton roots, also exposed from the public API.
// The public API exposed through std:vector's of joints can be used freely with
// the only restriction that the total number of joints does not exceed
// Skeleton::kMaxJoints.
struct RawSkeleton {
    // Declares the skeleton's roots. Can be empty if the skeleton has no joint.
    roots: Vec<Joint>,
}

impl RawSkeleton {
    // Construct an empty skeleton.
    pub fn new() -> RawSkeleton {
        return RawSkeleton {
            roots: vec![]
        };
    }

    // Tests for *this validity.
    // Returns true on success or false on failure if the number of joints exceeds
    // ozz::Skeleton::kMaxJoints.
    pub fn validate(&self) -> bool {
        if self.num_joints() > crate::skeleton::Constants::KMaxJoints as i32 {
            return false;
        }
        return true;
    }

    // Returns the number of joints of *this animation.
    // This function is not constant time as it iterates the hierarchy of joints
    // and counts them.
    pub fn num_joints(&self) -> i32 {
        todo!()
    }
}